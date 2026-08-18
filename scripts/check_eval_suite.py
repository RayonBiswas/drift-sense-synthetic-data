#!/usr/bin/env python3
"""
Verify the curated evaluation suite. Prints PASS/FAIL and the numbers behind it.

    python scripts/check_eval_suite.py

Three things are checked, because three things could silently be wrong:

  1. Image integrity   every case has both PNGs, 8-bit grayscale, search
                       exactly 1000x1000, reference within 100-256 px.
  2. Ground truth      the reference is de-rotated, rescaled to its annotated
                       footprint and correlated back into the search image
                       around the annotated centre. If the transform
                       bookkeeping were wrong the peak would land elsewhere.
                       A peak that lands a whole lattice pitch away is reported
                       separately -- that is aliasing, not a broken annotation.
  3. Axis isolation    within a sweep axis, the sub-seeds and every parameter
                       that is not under test must be byte-identical, and the
                       parameter under test must actually take distinct values.
                       Without this, an accuracy difference along an axis cannot
                       be attributed to that axis.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT / "dram_dataset"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dram_synth.qc import _gt_offset  # noqa: E402

SUITE = ROOT / "eval_suite"
SPLIT = "eval"

# Per axis: the field that must vary, and the fields that must not. A "vary"
# entry joined by "+" is a *combination* that must be distinct -- axis F is a
# 2x2 factorial of pitch and jitter plus a block-size pair, so each individual
# field takes only three values while all six combinations differ.
IMAGING = ["reference_imaging", "search_imaging"]
ARCHBLOCK = ["architecture"]
ISOLATION = {
    "A": {"vary": ["P09_scale"],
          "const": ["seeds", "P10_rotation_deg"] + ARCHBLOCK + IMAGING + ["center"]},
    "B": {"vary": ["reference_width"],
          "const": ["seeds", "P09_scale", "P10_rotation_deg"] + ARCHBLOCK
                   + IMAGING + ["center", "bbox"]},
    "C": {"vary": ["P10_rotation_deg"],
          "const": ["seeds", "P09_scale"] + ARCHBLOCK + IMAGING + ["center"]},
    "D": {"vary": ["search_imaging.noise_sigma"],
          "const": ["seeds", "P09_scale", "P10_rotation_deg"] + ARCHBLOCK
                   + ["reference_imaging", "center", "bbox"]},
    "E": {"vary": ["search_imaging"],
          "const": ["seeds", "P09_scale", "P10_rotation_deg"] + ARCHBLOCK
                   + ["reference_imaging"]},
    "F": {"vary": ["architecture.P01_word_line_pitch+architecture.P06_spacing_jitter"
                   "+architecture.block_size"],
          "const": ["seeds", "P09_scale", "P10_rotation_deg"] + IMAGING + ["center"]},
    # G is a set of scenarios, not a sweep; nothing to hold constant.
}


def dig(rec: dict, path: str):
    node = rec
    for part in path.split("."):
        node = node[part]
    return node


def main() -> int:
    ann = json.load(open(SUITE / SPLIT / "annotations.json", encoding="utf-8"))
    recs = ann["samples"]
    failures = []

    # ---- 1. image integrity ------------------------------------------------ #
    print(f"1. IMAGE INTEGRITY  ({len(recs)} cases)")
    bad = 0
    for r in recs:
        ref = cv2.imread(str(SUITE / SPLIT / r["reference_filename"]), cv2.IMREAD_UNCHANGED)
        sea = cv2.imread(str(SUITE / SPLIT / r["search_filename"]), cv2.IMREAD_UNCHANGED)
        why = []
        if ref is None or sea is None:
            why.append("unreadable")
        else:
            if ref.ndim != 2 or ref.dtype != np.uint8:
                why.append(f"reference not 8-bit gray ({ref.dtype}, {ref.ndim}d)")
            if sea.shape != (1000, 1000):
                why.append(f"search is {sea.shape}, expected (1000, 1000)")
            if not (100 <= ref.shape[0] <= 256):
                why.append(f"reference side {ref.shape[0]} outside 100-256")
            if ref.std() < 1.0 or sea.std() < 1.0:
                why.append("flat image")
        if why:
            bad += 1
            failures.append(f"{r['image_id']}: {'; '.join(why)}")
    print(f"   {len(recs) - bad}/{len(recs)} cases valid "
          f"(8-bit gray, search 1000x1000, reference 100-256 px, non-flat)")

    # ---- 2. ground truth --------------------------------------------------- #
    print("\n2. GROUND TRUTH  (reference correlated back to its annotated centre)")
    offs, aliased, unexplained, unprobed = [], [], [], []
    for r in recs:
        got = _gt_offset(SUITE, r)
        if got is None:
            unprobed.append(r["image_id"])
            continue
        arch = r["architecture"]
        word, bit = arch["P01_word_line_pitch"], arch["bit_line_pitch"]
        if got["offset"] > max(3.0 * word, 12.0):
            # An offset this large is only acceptable if it decomposes into a
            # whole number of lattice steps -- that is the reference matching
            # the right pattern at the wrong repeat, which is the phenomenon
            # under study. Anything else would mean the annotation is wrong.
            # The residual is allowed the sample's own scan-error budget, since
            # drift, vibration and raster shear bend the lattice locally.
            si = r["search_imaging"]
            slack = max(3.0, si["thermal_drift_px"] + si["vibration_amp_px"]
                        + 0.5 * si.get("raster_shear_px", 0.0))
            rx = (got["dx"] / bit - round(got["dx"] / bit)) * bit
            ry = (got["dy"] / word - round(got["dy"] / word)) * word
            explained = abs(rx) <= slack and abs(ry) <= slack
            aliased.append((r["image_id"], round(got["offset"], 1),
                            round(got["dx"] / bit, 2), round(got["dy"] / word, 2),
                            round(max(abs(rx), abs(ry)), 2), round(slack, 2),
                            explained))
            if not explained:
                unexplained.append(r["image_id"])
        offs.append(got["offset"])
    offs = np.array(offs)
    n_direct = len(offs) - len(aliased)
    ok = np.array([o for o in offs if o <= 12.0] or [0.0])
    print(f"   probed {len(offs)}/{len(recs)}   median offset {np.median(offs):.2f} px   "
          f"p95 {np.percentile(offs, 95):.2f} px")
    print(f"   landed on the annotated site: {n_direct}/{len(offs)}   "
          f"(median {np.median(ok):.2f} px on those)")
    print(f"   landed on a lattice repeat instead: {len(aliased)}")
    for cid, off, nx, ny, res, slack, ok_ in aliased:
        print(f"     {cid:16s} {off:6.1f} px = {nx:+.2f} bit-pitch, {ny:+.2f} "
              f"word-pitch  residual {res:.2f} px (budget {slack:.2f}) -> "
              f"{'lattice-explained' if ok_ else 'UNEXPLAINED'}")
    if unexplained:
        failures.append(f"aliased peaks not explained by the lattice: {unexplained}")
    if np.median(ok) > 3.0:
        failures.append(f"ground truth median offset {np.median(ok):.2f} px > 3 px")
    if n_direct < 0.75 * len(offs):
        failures.append(f"only {n_direct}/{len(offs)} cases correlate to the "
                        f"annotated site; annotation may be wrong")

    # ---- 3. axis isolation ------------------------------------------------- #
    print("\n3. AXIS ISOLATION  (one parameter moves, everything else pinned)")
    # A sweep is controlled *within* a replicate: across replicates the
    # specimen is meant to change, so grouping by axis alone would report every
    # pinned field as broken.
    by_group = {}
    for r in recs:
        by_group.setdefault((r["case"]["axis"], r["case"]["replicate"]), []).append(r)
    axes_seen = sorted({a for a, _ in by_group})
    for axis in axes_seen:
        reps = sorted(rep for a, rep in by_group if a == axis)
        rule = ISOLATION.get(axis)
        n_cases = sum(len(by_group[(axis, rep)]) for rep in reps)
        if rule is None:
            print(f"   axis {axis}  {n_cases} scenarios in {len(reps)} replicates "
                  f"-- not a sweep, no pin required")
            continue
        const_bad, vary_bad = [], []
        for rep in reps:
            group = by_group[(axis, rep)]
            for path in rule["const"]:
                vals = {json.dumps(dig(r, path), sort_keys=True) for r in group}
                if len(vals) != 1:
                    const_bad.append(f"r{rep}:{path}")
            for path in rule["vary"]:
                parts = path.split("+")
                vals = {json.dumps([dig(r, p) for p in parts], sort_keys=True)
                        for r in group}
                if len(vals) != len(group):
                    vary_bad.append(f"r{rep}:{path} ({len(vals)}/{len(group)})")
        # Replicates must differ from each other, or the suite is one specimen
        # copied five times.
        fingerprints = {json.dumps(by_group[(axis, rep)][0]["architecture"],
                                   sort_keys=True) for rep in reps}
        if len(fingerprints) != len(reps) and axis != "F":
            const_bad.append(f"only {len(fingerprints)}/{len(reps)} distinct specimens")
        state = "ok" if not const_bad and not vary_bad else "BROKEN"
        print(f"   axis {axis}  {n_cases} cases  {len(reps)} replicates x "
              f"{n_cases // len(reps)} levels  {len(rule['const'])} fields pinned "
              f"per sweep  -> {state}")
        if const_bad:
            failures.append(f"axis {axis}: these should be constant but vary: {const_bad}")
        if vary_bad:
            failures.append(f"axis {axis}: these should sweep but do not: {vary_bad}")

    print("\n" + "=" * 60)
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"RESULT: PASS  ({len(recs)} cases, {len(axes_seen)} axes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

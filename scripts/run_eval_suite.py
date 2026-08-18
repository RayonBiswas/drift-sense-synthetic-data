#!/usr/bin/env python3
"""
Run the localizers over the curated evaluation suite and report per axis.

    python scripts/run_eval_suite.py

Scoring is the Euclidean distance in search pixels between the returned centre
and the annotated centre. Two methods are run on identical inputs:

  v1  dram_dataset/localize.py     ZNCC size/angle sweep + centre-weighted
                                   tie-break over rival peaks
  v2  dram_dataset/localize_v2.py  the same, plus a sub-pixel parabolic peak fit
                                   and a continuous centre prior

Per axis the script prints the error against the swept level, which is what
makes the suite worth having: a rising error along one axis names the cause.
It also reports, per axis, whether the rival-peak count would have flagged the
failure -- a localizer that knows when it is lost can ask for a re-image
instead of writing to the wrong site.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT / "dram_dataset"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

TOL = 5.0  # "correct" means within 5 px of the annotated centre


def run_one(job: dict) -> dict:
    from localize import localize
    from localize_v2 import localize_v2

    ref = cv2.imread(job["ref"], cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(job["search"], cv2.IMREAD_GRAYSCALE)
    gx, gy = job["gx"], job["gy"]
    out = {k: job[k] for k in ("case_id", "axis", "axis_label", "level",
                               "replicate", "probes", "difficulty",
                               "architecture_kind", "gx", "gy",
                               "footprint_px", "pitch")}

    t = time.time()
    r1 = localize(ref, search)
    out["v1"] = {"x": r1.center_x, "y": r1.center_y,
                 "err": float(np.hypot(r1.center_x - gx, r1.center_y - gy)),
                 "score": float(r1.score), "rivals": int(r1.n_candidates),
                 "size": int(r1.size), "angle": float(r1.angle),
                 "tie_break": r1.tie_break_applied, "sec": time.time() - t}

    t = time.time()
    r2 = localize_v2(ref, search)
    out["v2"] = {"x": r2.center_x, "y": r2.center_y,
                 "err": float(np.hypot(r2.center_x - gx, r2.center_y - gy)),
                 "score": float(r2.score), "rivals": int(r2.n_candidates),
                 "psr": float(r2.psr), "env": float(r2.envelope_agreement),
                 "size": int(r2.size), "angle": float(r2.angle),
                 "sec": time.time() - t}

    # How far the answer is in units of the lattice pitch. An error of a whole
    # number of pitches is the wrong repeat of the right pattern -- a different
    # failure from a matcher that simply did not find anything.
    for v in ("v1", "v2"):
        out[v]["err_in_pitches"] = round(out[v]["err"] / out["pitch"], 2)
    return out


def summarize(rows, key):
    err = np.array([r[key]["err"] for r in rows])
    return {"n": len(rows),
            "acc@1px": round(100 * float((err <= 1).mean()), 1),
            "acc@5px": round(100 * float((err <= TOL).mean()), 1),
            "acc@10px": round(100 * float((err <= 10).mean()), 1),
            "median_err_px": round(float(np.median(err)), 2),
            "max_err_px": round(float(err.max()), 1),
            "mean_sec": round(float(np.mean([r[key]["sec"] for r in rows])), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="eval_suite")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="reports/eval_suite")
    args = ap.parse_args()

    root = ROOT / args.suite / "eval"
    ann = json.load(open(root / "annotations.json", encoding="utf-8"))
    recs = ann["samples"]

    jobs = [{"ref": str(root / r["reference_filename"]),
             "search": str(root / r["search_filename"]),
             "gx": r["center"]["x"], "gy": r["center"]["y"],
             "case_id": r["case"]["case_id"], "axis": r["case"]["axis"],
             "axis_label": r["case"]["axis_label"], "level": r["case"]["level"],
             "replicate": r["case"]["replicate"],
             "probes": r["case"]["probes"], "difficulty": r["difficulty"],
             "architecture_kind": r.get("architecture_kind", "dram"),
             "footprint_px": r["footprint_px"],
             "pitch": r["architecture"]["P01_word_line_pitch"]} for r in recs]

    print(f"running v1 and v2 over {len(jobs)} curated cases "
          f"({args.workers} workers)")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run_one, jobs))
    print(f"done in {time.time() - t0:.0f}s\n")

    order = {r["case_id"]: i for i, r in enumerate(jobs)}
    rows.sort(key=lambda r: order[r["case_id"]])

    # ---- per-level table --------------------------------------------------- #
    # Each level is now five independent specimens, so the number below is a
    # rate rather than one specimen's luck. This is the table that says which
    # stressor actually costs accuracy.
    print(f"{'axis':5s} {'level':26s} {'n':>2s} {'v1 acc@5px':>10s} "
          f"{'v2 acc@5px':>10s} {'v2 med err':>10s} {'v2 p90 err':>10s} "
          f"{'rivals':>7s} {'lattice':>8s}")
    print("-" * 96)
    by_level, level_order = {}, []
    for r in rows:
        key = (r["axis"], str(r["level"]))
        if key not in by_level:
            by_level[key] = []
            level_order.append(key)
        by_level[key].append(r)

    level_table = {}
    last_axis = None
    for key in level_order:
        g = by_level[key]
        if key[0] != last_axis:
            if last_axis is not None:
                print()
            print(f"[axis {key[0]}] {g[0]['axis_label']}")
            last_axis = key[0]
        e1 = np.array([r["v1"]["err"] for r in g])
        e2 = np.array([r["v2"]["err"] for r in g])
        riv = np.mean([r["v2"]["rivals"] for r in g])
        # Share of this level's misses that sit a whole number of lattice steps
        # away -- i.e. right pattern, wrong repeat.
        miss = [r for r in g if r["v2"]["err"] > TOL]
        lat = [r for r in miss
               if r["v2"]["err_in_pitches"] >= 1.0
               and abs(r["v2"]["err_in_pitches"] - round(r["v2"]["err_in_pitches"])) < 0.25]
        entry = {"axis": key[0], "level": key[1], "n": len(g),
                 "v1_acc@5px": round(100 * float((e1 <= TOL).mean()), 1),
                 "v2_acc@5px": round(100 * float((e2 <= TOL).mean()), 1),
                 "v2_median_err_px": round(float(np.median(e2)), 2),
                 "v2_p90_err_px": round(float(np.percentile(e2, 90)), 1),
                 "mean_rivals": round(float(riv), 1),
                 "misses": len(miss), "lattice_misses": len(lat)}
        level_table[f"{key[0]}|{key[1]}"] = entry
        print(f"{'':5s} {key[1][:26]:26s} {len(g):2d} "
              f"{entry['v1_acc@5px']:9.1f}% {entry['v2_acc@5px']:9.1f}% "
              f"{entry['v2_median_err_px']:10.2f} {entry['v2_p90_err_px']:10.1f} "
              f"{riv:7.1f} {str(len(lat)) + '/' + str(len(miss)):>8s}")

    # ---- per-axis summary -------------------------------------------------- #
    by_axis = {}
    for r in rows:
        by_axis.setdefault(r["axis"], []).append(r)

    print(f"\n{'axis':6s} {'what it varies':34s} {'n':>3s} "
          f"{'v1 acc@5px':>11s} {'v1 med':>8s} {'v2 acc@5px':>11s} {'v2 med':>8s}")
    print("-" * 88)
    axis_table = {}
    for ax in sorted(by_axis):
        g = by_axis[ax]
        s1, s2 = summarize(g, "v1"), summarize(g, "v2")
        axis_table[ax] = {"label": g[0]["axis_label"], "v1": s1, "v2": s2}
        print(f"{ax:6s} {g[0]['axis_label'][:34]:34s} {len(g):3d} "
              f"{s1['acc@5px']:10.1f}% {s1['median_err_px']:8.2f} "
              f"{s2['acc@5px']:10.1f}% {s2['median_err_px']:8.2f}")
    s1, s2 = summarize(rows, "v1"), summarize(rows, "v2")
    print("-" * 88)
    print(f"{'ALL':6s} {'whole curated suite':34s} {len(rows):3d} "
          f"{s1['acc@5px']:10.1f}% {s1['median_err_px']:8.2f} "
          f"{s2['acc@5px']:10.1f}% {s2['median_err_px']:8.2f}")

    # ---- confidence gating ------------------------------------------------- #
    # The operational question is not "how often is it right" but "when it is
    # wrong, does it know". rivals == 1 means the correlation surface had a
    # single dominant peak; rivals > 1 means the method itself saw ambiguity.
    print("\nCONFIDENCE GATING on the curated suite (v2)")
    err = np.array([r["v2"]["err"] for r in rows])
    riv = np.array([r["v2"]["rivals"] for r in rows])
    gating = {}
    print(f"{'rule':16s} {'kept':>6s} {'acc@5px kept':>13s} "
          f"{'acc@5px dropped':>16s} {'median kept':>12s}")
    print("-" * 68)
    for thr, label in ((1, "rivals == 1"), (2, "rivals <= 2"), (4, "rivals <= 4")):
        keep = riv <= thr
        if keep.sum() == 0:
            continue
        ak = 100 * float((err[keep] <= TOL).mean())
        ad = (100 * float((err[~keep] <= TOL).mean())
              if (~keep).sum() else float("nan"))
        mk = float(np.median(err[keep]))
        gating[label] = {"kept": int(keep.sum()), "acc@5px_kept": round(ak, 1),
                         "acc@5px_dropped": round(ad, 1),
                         "median_kept": round(mk, 2)}
        print(f"{label:16s} {keep.sum():6d} {ak:12.1f}% {ad:15.1f}% {mk:12.2f}")

    # ---- how the failures fail --------------------------------------------- #
    # A wrong answer that sits a whole number of lattice pitches from the truth
    # is periodic confusion. Anything else is a matcher that lost the pattern.
    wrong = [r for r in rows if r["v2"]["err"] > TOL]
    lattice = [r for r in wrong
               if abs(r["v2"]["err_in_pitches"] - round(r["v2"]["err_in_pitches"])) < 0.25
               and r["v2"]["err_in_pitches"] >= 1.0]
    print(f"\nFAILURE MODE (v2): {len(wrong)}/{len(rows)} cases outside {TOL:.0f} px; "
          f"{len(lattice)} of them sit within a quarter-pitch of a whole number "
          f"of lattice steps (periodic confusion rather than a lost pattern)")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"suite": args.suite, "n_cases": len(rows), "tolerance_px": TOL,
                   "overall": {"v1": s1, "v2": s2}, "per_axis": axis_table,
                   "per_level": level_table,
                   "gating_v2": gating, "cases": rows}, f, indent=2)
    with open(out_dir / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "axis", "axis_label", "level", "replicate",
                    "difficulty",
                    "architecture", "gt_x", "gt_y", "v1_x", "v1_y", "v1_err_px",
                    "v1_rivals", "v2_x", "v2_y", "v2_err_px", "v2_rivals",
                    "v2_psr", "v2_err_in_pitches", "probes"])
        for r in rows:
            w.writerow([r["case_id"], r["axis"], r["axis_label"], r["level"],
                        r["replicate"], r["difficulty"], r["architecture_kind"],
                        round(r["gx"], 2), round(r["gy"], 2),
                        round(r["v1"]["x"], 2), round(r["v1"]["y"], 2),
                        round(r["v1"]["err"], 2), r["v1"]["rivals"],
                        round(r["v2"]["x"], 2), round(r["v2"]["y"], 2),
                        round(r["v2"]["err"], 2), r["v2"]["rivals"],
                        round(r["v2"]["psr"], 2), r["v2"]["err_in_pitches"],
                        r["probes"]])
    print(f"\nwrote {args.out}/report.json and {args.out}/results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

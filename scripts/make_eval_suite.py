#!/usr/bin/env python3
"""
Build the curated evaluation suite: 420 controlled test cases, seven axes.

    python scripts/make_eval_suite.py --output eval_suite

The 1000-pair dataset is a *sample* of the joint parameter distribution: every
pair varies every parameter at once, so a failure there cannot be attributed to
any one cause. This suite is the complement. Within one *replicate* of an axis
every case shares a base seed, so the specimen, the placement and both captures
are identical and **exactly one parameter moves**. A difference in localization
error along an axis is therefore caused by that axis and nothing else.

Each sweep is then repeated over REPLICATES independent specimens, drawn with
their own layout, imaging and stage placement. Without that, every level of an
axis has a sample size of one, and the axis reports whichever specimen its seed
happened to draw rather than the effect under test.

Each case carries a `probes` string saying what property of the algorithm it
tests. That is the point of the suite -- not volume, but attributable evidence.

Axes
  A  scale             P09, the reference footprint in search pixels
  B  resolution ratio  stored reference px per search px (the magnification gap)
  C  rotation          P10, including two cases outside the matcher's sweep
  D  noise / dose      P16, the detector and shot-noise budget of the search scan
  E  blur / scan error P11, P19, P20 and the scan-linearity terms
  F  periodicity       P01, P05, P06 -- how self-similar the layout is
  G  operational       combined worst cases, placement faults, cross-family
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dram_dataset"))

import numpy as np  # noqa: E402

from dram_synth.sample import annotate, build_sample  # noqa: E402

SPLIT = "eval"

# How many independent specimens each sweep is repeated over. Ten gives every
# level a sample size of ten, so an axis result is a rate readable to 10% rather
# than one specimen's luck, and the whole suite still renders and scores in
# minutes on CPU.
REPLICATES = 10

AXES = "ABCDEFG"


def seed_for(rep: int) -> int:
    """The base specimen for a replicate.

    Deliberately independent of the axis: every axis sweeps the *same* ten
    specimens. If each axis drew its own, a difference between two axes would
    mostly report which specimens each happened to draw, and the natural
    reading of the results table -- "which stressor costs the most accuracy" --
    would be wrong. Sharing the specimens makes the axes comparable to each
    other as well as internally controlled.
    """
    return 11_000_000 + rep


def pinned_position(rep: int) -> dict:
    """Placement for one replicate: pinned within a sweep, varied across
    replicates, and shared across axes for the same reason the seed is.

    Pinning is what makes a sweep controlled -- every level must be scored at
    the same distance from the frame centre, or the centre prior would help
    some levels more than others. Varying it across replicates is what stops
    the suite from measuring one lucky offset: the draw uses the same truncated
    Gaussian (sigma 110 px per axis) that models real stage positioning error.
    """
    rng = np.random.default_rng(900_000 + rep)
    x, y = (float(np.clip(500.0 + rng.normal(0.0, 110.0), 90.0, 910.0))
            for _ in range(2))
    return {"position_x": x, "position_y": y, "position_mode": "drift"}


def _case(cid, axis, axis_label, level, seed, difficulty, overrides, probes,
          architecture="dram"):
    return {"case_id": cid, "axis": axis, "axis_label": axis_label,
            "level": level, "sample_seed": seed, "difficulty": difficulty,
            "architecture": architecture, "overrides": overrides,
            "probes": probes}


def build_cases() -> list:
    """Every case in the suite: seven axes, each swept over REPLICATES specimens."""
    cases = []
    for rep in range(1, REPLICATES + 1):
        cases += _axis_a(rep) + _axis_b(rep) + _axis_c(rep)
        cases += _axis_d(rep) + _axis_e(rep) + _axis_f(rep) + _axis_g(rep)
    return cases


# ---------------------------------------------------------------------- A: scale
# The matcher does not know the true footprint; it sweeps template sizes 80..125
# px in 5 px steps. This axis asks how much error is simply scale quantisation,
# and whether accuracy is symmetric about 1.0.
def _axis_a(rep: int) -> list:
    seed, pos = seed_for(rep), pinned_position(rep)
    return [_case(
        f"A_scale_{int(scale*100):03d}_r{rep}", "A", "scale (P09)", scale,
        seed, "medium", {**pos, "scale": scale, "rotation_deg": 0.0},
        f"Footprint is {scale*100:.0f} px. Tests whether the size sweep recovers "
        f"an unknown footprint, and how a residual scale error between template "
        f"and target degrades the ZNCC peak.")
        for scale in (0.85, 0.90, 0.95, 1.05, 1.10, 1.15)]


# ----------------------------------------------------------- B: resolution ratio
# The stored reference is a higher-magnification capture: the same 100 px of
# device is stored in 100..256 px. The algorithm must resample it down by an
# unknown factor. This is the magnification gap the problem statement means,
# isolated from every other effect (scale and rotation are pinned).
def _axis_b(rep: int) -> list:
    seed, pos = seed_for(rep), pinned_position(rep)
    return [_case(
        f"B_refpx_{ref_px}_r{rep}", "B", "reference resolution ratio", ref_px,
        seed, "medium",
        {**pos, "scale": 1.0, "rotation_deg": 0.0, "reference_size_px": ref_px},
        f"Reference stored at {ref_px} px for a 100 px footprint = "
        f"{ref_px/100.0:.2f}x more stored resolution than the search shows. "
        f"Tests the down-resampling path: how much fine detail survives "
        f"INTER_AREA reduction to the search's pixel scale, and whether a larger "
        f"stored reference helps or only adds detail the search cannot resolve.")
        for ref_px in (100, 128, 160, 192, 224, 256)]


# ------------------------------------------------------------------- C: rotation
# 0 to 5 deg is inside the dataset's own range and inside the matcher's +-5 deg
# sweep. 7 and 10 deg are deliberately outside both: they measure graceful
# degradation, and whether the confidence signal notices the template no longer
# fits.
def _axis_c(rep: int) -> list:
    seed, pos = seed_for(rep), pinned_position(rep)
    return [_case(
        f"C_rot_{str(rot).replace('.', 'p')}_r{rep}", "C", "rotation (P10)", rot,
        seed, "medium", {**pos, "rotation_deg": rot, "scale": 1.0},
        f"Reference rotated {rot} deg against the search frame. The matcher "
        f"sweeps -5..+5 deg in 2.5 deg steps, so this tests angular quantisation "
        f"up to 5 deg and out-of-sweep behaviour at 7 and 10 deg.")
        for rot in (0.0, 1.5, 3.0, 5.0, 7.0, 10.0)]


# --------------------------------------------------------------- D: noise / dose
# ZNCC subtracts the local mean and divides by the local standard deviation, so
# noise should lower the peak height without moving it -- until the noise floor
# swamps the ~0.02 ZNCC gap between the true peak and its lattice rivals, at
# which point the answer jumps a whole repeat. This axis finds that breakpoint.
def _axis_d(rep: int) -> list:
    seed, pos = seed_for(rep), pinned_position(rep)
    return [_case(
        f"D_noise_s{int(sigma):02d}_r{rep}", "D", "search noise sigma (P16)",
        sigma, seed, "medium",
        {**pos, "scale": 1.0, "rotation_deg": 0.0,
         "search_imaging": {"noise_sigma": sigma, "poisson_dose": float(dose)}},
        f"Search capture at detector sigma {sigma} DN and {dose} e- dose; the "
        f"reference capture is untouched. Tests ZNCC's claimed invariance to "
        f"additive noise, and locates the SNR at which the noise floor exceeds "
        f"the score gap between the true peak and its rivals.")
        for sigma, dose in ((2.0, 1500), (5.0, 700), (9.0, 300),
                            (13.0, 150), (16.0, 80), (22.0, 40))]


# ------------------------------------------------------- E: blur and scan error
# The terms that move or smear geometry rather than just intensity. Blur matters
# most: it destroys the sub-pixel placement jitter that is the only thing
# distinguishing one lattice repeat from the next.
E_SPECS = [
    ("blur_0p5", "blur sigma 0.5 px", {"blur_sigma": 0.5},
     "Near-diffraction-limited search scan. The control point for this axis: "
     "fine lattice detail is intact, so any error here is not an optics problem."),
    ("blur_1p2", "blur sigma 1.2 px", {"blur_sigma": 1.2},
     "Typical fast wide-area scan. Tests how much lattice contrast the matcher "
     "needs before periodic repeats become indistinguishable."),
    ("blur_2p4", "blur sigma 2.4 px", {"blur_sigma": 2.4},
     "Worst blur in the dataset. Directly tests the project's central claim that "
     "blur, not noise, is what makes repeats genuinely identical -- an "
     "information loss no matcher can undo."),
    ("drift_5px", "thermal drift 5 px/frame", {"thermal_drift_px": 5.0},
     "Maximum stage drift accumulated across one raster. The target is displaced "
     "non-rigidly, so a rigid template cannot fit it exactly; ground truth "
     "follows the same warp, so the residual is real and not an annotation error."),
    ("vib_1p8", "vibration 1.8 px", {"vibration_amp_px": 1.8},
     "Maximum stage/column vibration: rows are displaced quasi-periodically. "
     "Tests robustness to a distortion that is itself periodic, and can therefore "
     "manufacture correlation peaks of its own."),
    ("scanlin", "barrel 0.03 + shear 4.5 px",
     {"barrel_k": 0.03, "raster_shear_px": 4.5},
     "Scan-linearity error: radial barrel distortion plus progressive raster "
     "shear. Position-dependent geometry, so one template fits differently at "
     "different places in the frame -- the failure a single global scale and "
     "rotation sweep cannot represent."),
]


def _axis_e(rep: int) -> list:
    seed, pos = seed_for(rep), pinned_position(rep)
    return [_case(
        f"E_{slug}_r{rep}", "E", "blur and scan error (P11/P19/P20)", label,
        seed, "medium",
        {**pos, "scale": 1.0, "rotation_deg": 0.0, "search_imaging": ov}, probes)
        for slug, label, ov, probes in E_SPECS]


# ---------------------------------------------------------------- F: periodicity
# The heart of the problem statement. Repeat count inside the footprint is
# 100/pitch; spacing jitter is the only thing making one repeat physically
# distinguishable from the next; block size sets how much larger-scale layout
# context is visible at all.
F_SPECS = [
    ("fine_rigid", "pitch 6.0, jitter 0.02",
     {"word_line_pitch": 6.0, "spacing_jitter": 0.02},
     "~16 word-line repeats inside the footprint and a near-perfect lattice. The "
     "maximum-ambiguity case: many correlation peaks separated by ~0.02 ZNCC. "
     "This is the case the whole problem statement is about."),
    ("fine_jitter", "pitch 6.0, jitter 0.35",
     {"word_line_pitch": 6.0, "spacing_jitter": 0.35},
     "The same 16 repeats, but each line individually displaced. Isolates "
     "placement jitter as the disambiguating signal: if accuracy rises against "
     "fine_rigid, the matcher is using jitter rather than the lattice itself."),
    ("coarse_rigid", "pitch 14.0, jitter 0.02",
     {"word_line_pitch": 14.0, "spacing_jitter": 0.02},
     "~7 coarse repeats, near-perfect lattice. Fewer rivals per footprint, but "
     "each is a larger displacement, so a wrong pick costs more pixels. Tests "
     "whether error magnitude tracks pitch as lattice theory predicts."),
    ("coarse_jitter", "pitch 14.0, jitter 0.35",
     {"word_line_pitch": 14.0, "spacing_jitter": 0.35},
     "Coarse pitch with strong jitter -- the least ambiguous layout in the suite. "
     "Establishes the accuracy ceiling the method reaches when periodic ambiguity "
     "is removed but every imaging effect is still present."),
    ("many_mats", "block 90 (many mats)",
     {"block_size": 90.0, "strip_width": 18.0},
     "Small array blocks, so several mat boundaries and routing strips fall "
     "inside the frame. Tests whether larger-scale layout context can break the "
     "tie between repeats -- the hypothesis the v2 envelope stage was built on "
     "and which measurement rejected."),
    ("one_mat", "block 320 (single mat)",
     {"block_size": 320.0, "strip_width": 6.0},
     "One large uninterrupted array: the footprint and its neighbourhood are pure "
     "lattice with no unique landmark anywhere near them. The complement of "
     "many_mats, and the honest worst case for any context-based method."),
]


def _axis_f(rep: int) -> list:
    seed, pos = seed_for(rep), pinned_position(rep)
    return [_case(
        f"F_{slug}_r{rep}", "F", "periodicity (P01/P05/P06)", label, seed,
        "medium", {**pos, "scale": 1.0, "rotation_deg": 0.0, **ov}, probes)
        for slug, label, ov, probes in F_SPECS]


# ---------------------------------------------------------------- G: operational
# Combined and adversarial cases. Not controlled sweeps -- each is a scenario the
# tool actually meets on the floor.
HARD_STACK = {
    "scale": 0.85, "rotation_deg": 5.0, "spacing_jitter": 0.02,
    "word_line_pitch": 6.5, "defect_density": 0.08,
    "search_imaging": {"blur_sigma": 2.4, "noise_sigma": 16.0,
                       "poisson_dose": 60.0, "thermal_drift_px": 5.0,
                       "vibration_amp_px": 1.8},
}


def _axis_g(rep: int) -> list:
    seed, pos = seed_for(rep), pinned_position(rep)
    # Placement faults get their own positions, jittered per replicate so the
    # scenario is not measured at one lucky offset either.
    rng = np.random.default_rng(770_000 + rep)
    lost = {"position_x": float(rng.uniform(120.0, 240.0)),
            "position_y": float(rng.uniform(120.0, 240.0)),
            "position_mode": "uniform"}
    edge = {"position_x": float(rng.uniform(78.0, 95.0)),
            "position_y": float(rng.uniform(300.0, 700.0)),
            "position_mode": "uniform"}
    specs = [
        ("worst_case", "all stressors at maximum", "hard", "dram",
         {**pos, **HARD_STACK},
         "Every stressor at its dataset maximum at once: 0.85x scale, 5 deg "
         "rotation, a near-perfect fine lattice, 2.4 px blur, sigma 16 noise, "
         "60 e- dose, full drift and vibration. The floor of the method -- if "
         "the confidence signal does not flag this case, gating is not "
         "trustworthy."),
        ("site_lost", "uniform placement, corner", "medium", "dram",
         {**lost, "scale": 1.0, "rotation_deg": 0.0},
         "The stage lost the site entirely and re-acquired it near a corner, "
         "some 400 px from the frame centre. The centre prior is now actively "
         "misleading, so this tests that the prior weights the decision rather "
         "than overriding it -- the exact failure the hard 'center' tie-break "
         "policy exhibits."),
        ("frame_edge", "target at the frame margin", "medium", "dram",
         {**edge, "scale": 1.0, "rotation_deg": 0.0},
         "Target pressed against the left margin, where vignetting is strongest "
         "and part of its surrounding context lies outside the frame. Tests "
         "boundary handling and whether radial shading biases ZNCC."),
        ("defect_heavy", "defect density 0.08", "hard", "dram",
         {**pos, "defect_density": 0.08, "scale": 1.0, "rotation_deg": 0.0},
         "Maximum defect density: bridges, breaks and missing contacts. Defects "
         "are unique landmarks, so they should *help* -- but the reference and "
         "the search image them under different scan errors, so this tests "
         "whether the matcher can exploit a landmark it sees twice, imperfectly."),
        ("finfet", "FinFET, nominal", "medium", "finfet",
         {**pos, "scale": 1.0, "rotation_deg": 0.0},
         "A different device family -- fins crossed by gate stripes, contacts in "
         "the diffusion -- through an identical imaging chain. Tests that the "
         "method is not tuned to DRAM's particular duty cycle and contact grid."),
        ("finfet_hard", "FinFET, all stressors", "hard", "finfet",
         {**pos, **HARD_STACK},
         "The worst case repeated on FinFET. Comparing it against worst_case "
         "separates 'this layout is hard' from 'this imaging is hard', which a "
         "single-architecture suite cannot do."),
    ]
    return [_case(f"G_{slug}_r{rep}", "G", "operational scenarios", label, seed,
                  diff, ov, probes, architecture=arch)
            for slug, label, diff, arch, ov, probes in specs]


def render_case(job: dict) -> dict:
    """Render one case and write both PNGs. Module level for multiprocessing."""
    import cv2

    case = job["case"]
    sample = build_sample(case["sample_seed"], case["difficulty"],
                          job["supersample"], case["architecture"],
                          case["overrides"])

    name = f"{case['case_id']}.png"
    flags = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if not cv2.imwrite(os.path.join(job["reference_dir"], name),
                       sample["reference_img"], flags):
        raise IOError(f"failed to write reference for {case['case_id']}")
    if not cv2.imwrite(os.path.join(job["search_dir"], name),
                       sample["search_img"], flags):
        raise IOError(f"failed to write search for {case['case_id']}")

    record = annotate(sample, case["case_id"], SPLIT,
                      f"references/{name}", f"searches/{name}")
    record.pop("_stats", None)
    record["case"] = {k: case[k] for k in
                      ("case_id", "axis", "axis_label", "level", "probes")}
    record["case"]["replicate"] = int(case["case_id"].rsplit("_r", 1)[1])
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="eval_suite")
    ap.add_argument("--supersample", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    root = Path(ROOT) / args.output
    (root / SPLIT / "references").mkdir(parents=True, exist_ok=True)
    (root / SPLIT / "searches").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)

    cases = build_cases()
    workers = args.workers or max(1, min((os.cpu_count() or 2) - 1, 6))
    print(f"rendering {len(cases)} curated cases into {root} "
          f"({workers} workers, supersample {args.supersample}x)")

    jobs = [{"case": c, "supersample": args.supersample,
             "reference_dir": str(root / SPLIT / "references"),
             "search_dir": str(root / SPLIT / "searches")} for c in cases]

    started = time.time()
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(render_case, j): j for j in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            out[job["case"]["case_id"]] = fut.result()
            print(f"\r  {n}/{len(jobs)}", end="", flush=True)
    records = [out[c["case_id"]] for c in cases]
    print(f"\n  done in {time.time() - started:.0f}s")

    axes = {}
    for c in cases:
        axes.setdefault(c["axis"], {"label": c["axis_label"], "n": 0})["n"] += 1

    with open(root / SPLIT / "annotations.json", "w", encoding="utf-8") as f:
        json.dump({"split": SPLIT, "count": len(records),
                   "search_image_size": [1000, 1000],
                   "coordinate_frame": (
                       "Pixels of the final 1000x1000 search image, origin "
                       "top-left, x right, y down. `center` is the answer the "
                       "algorithm must return."),
                   "axes": axes, "samples": records}, f, indent=2)

    with open(root / "metadata" / "suite_config.json", "w", encoding="utf-8") as f:
        json.dump({"generator": "dram_synth", "suite": "curated_eval_suite",
                   "n_cases": len(cases), "axes": axes,
                   "supersample_factor": args.supersample,
                   "replicates": REPLICATES,
                   "base_seeds": {f"r{r}": seed_for(r)
                                  for r in range(1, REPLICATES + 1)},
                   "pinned_placement": {f"r{r}": pinned_position(r)
                                        for r in range(1, REPLICATES + 1)},
                   "shared_specimens_across_axes": True,
                   "reproduce_command":
                       f"python scripts/make_eval_suite.py --output {args.output}",
                   "cases": cases}, f, indent=2)

    print(f"  wrote {root/SPLIT/'annotations.json'} and metadata/suite_config.json")
    for ax in sorted(axes):
        print(f"    axis {ax}  {axes[ax]['n']:2d} cases  {axes[ax]['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

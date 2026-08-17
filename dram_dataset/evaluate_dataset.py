#!/usr/bin/env python3
"""
Baseline evaluation on the generated dataset.

    python evaluate.py --dataset dataset                    # validation + test
    python evaluate.py --dataset dataset --splits test      # one split
    python evaluate.py --dataset dataset --splits train validation test --limit 200

What this measures
------------------
The baseline is **ZNCC template matching** -- zero-mean normalized
cross-correlation, which is what `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`
computes. It is a *classical, training-free* matcher: there is no model and no
training step, so there is no train/test fitting to report. What is reported is
localization accuracy on the held-out splits, which is the number that says how
hard the dataset actually is.

The matcher is given no privileged information. It does not read the annotated
scale, rotation or position; it sweeps a range of template sizes and rotations
and keeps the highest-correlating one. Ground truth is used only to score the
answer.

Metrics
-------
  accuracy@N px   fraction of samples whose predicted centre lands within N px
                  of the true centre
  median error    median distance from prediction to truth, in search pixels
  AP              area under the precision-recall curve, sweeping the ZNCC score
                  as an acceptance threshold. Every sample has exactly one true
                  match, so total positives = N and recall = TP / N.
  lattice-explained
                  a DRAM array is periodic, so a wrong answer is often the
                  *right pattern one repeat over*. Errors that resolve to a whole
                  number of word-line / bit-line pitches are counted separately,
                  because they say something quite different about the matcher
                  than a random miss does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

# Template sizes swept, in search pixels. The true footprint is 100 * scale with
# scale in [0.85, 1.15], and barrel distortion shrinks it slightly further.
DEFAULT_SIZES = list(range(80, 126, 5))
# Rotations swept. The reference sits at +-5 deg relative to the search.
DEFAULT_ANGLES = [-5.0, -2.5, 0.0, 2.5, 5.0]
# Fraction of the template kept after rotation, to drop undefined corners.
INNER_FRACTION = 0.78


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--splits", nargs="+", default=["validation", "test"],
                   choices=["train", "validation", "test"])
    p.add_argument("--tolerance-px", type=float, default=10.0,
                   help="distance within which a prediction counts as correct")
    p.add_argument("--limit", type=int, default=0,
                   help="evaluate at most this many samples per split (0 = all)")
    p.add_argument("--workers", type=int, default=0,
                   help="worker processes; 0 = auto, 1 = serial")
    p.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="template sizes to sweep (default 80..125 step 5)")
    p.add_argument("--angles", type=float, nargs="+", default=None,
                   help="rotations to sweep in degrees (default -5..5 step 2.5)")
    p.add_argument("--output", default="evaluation", help="where to write results")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def zncc_locate(reference: np.ndarray, search: np.ndarray,
                sizes, angles) -> dict:
    """Multi-scale, multi-rotation ZNCC search over the whole search image.

    Returns the highest-scoring candidate: its centre in search pixels, the ZNCC
    score in [-1, 1], and the size and rotation that produced it.
    """
    best = None
    sh, sw = search.shape
    for size in sizes:
        if size < 16:
            continue
        base = cv2.resize(reference, (size, size), interpolation=cv2.INTER_AREA)
        k = max(int(size * INNER_FRACTION), 8)
        off = (size - k) // 2
        for angle in angles:
            if angle != 0.0:
                m = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, 1.0)
                rotated = cv2.warpAffine(base, m, (size, size), flags=cv2.INTER_LINEAR)
            else:
                rotated = base
            tmpl = rotated[off:off + k, off:off + k]
            if k >= sh or k >= sw:
                continue
            result = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "x": float(loc[0] + k / 2.0),
                    "y": float(loc[1] + k / 2.0),
                    "size": int(size),
                    "angle": float(angle),
                }
    return best


def _lattice_explained(dx: float, dy: float, record: dict,
                       max_steps: int = 6, residual_px: float = 3.0) -> bool:
    """Is the error a whole number of lattice repeats?"""
    a = record["architecture"]
    px, py = a["bit_line_pitch"], a["P01_word_line_pitch"]
    if px <= 0 or py <= 0:
        return False
    nx, ny = round(dx / px), round(dy / py)
    if (nx, ny) == (0, 0) or abs(nx) > max_steps or abs(ny) > max_steps:
        return False
    return bool(np.hypot(dx - nx * px, dy - ny * py) <= residual_px)


def evaluate_one(job: dict) -> dict:
    """Score a single sample. Module-level for Windows multiprocessing."""
    root = Path(job["root"])
    rec = job["record"]
    base = root / rec["split"]
    reference = cv2.imread(str(base / rec["reference_filename"]), cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(str(base / rec["search_filename"]), cv2.IMREAD_GRAYSCALE)
    if reference is None or search is None:
        raise IOError(f"could not read images for {rec['image_id']}")

    started = time.time()
    match = zncc_locate(reference, search, job["sizes"], job["angles"])
    elapsed = time.time() - started

    gt_x, gt_y = rec["center"]["x"], rec["center"]["y"]
    dx, dy = match["x"] - gt_x, match["y"] - gt_y
    error = float(np.hypot(dx, dy))

    return {
        "image_id": rec["image_id"],
        "split": rec["split"],
        "difficulty": rec["difficulty"],
        "score": match["score"],
        "pred_x": match["x"], "pred_y": match["y"],
        "gt_x": gt_x, "gt_y": gt_y,
        "error_px": error,
        "dx": float(dx), "dy": float(dy),
        "pred_size": match["size"], "pred_angle": match["angle"],
        "true_footprint": rec["footprint_px"],
        "true_rotation": rec["P10_rotation_deg"],
        "lattice_explained": _lattice_explained(dx, dy, rec),
        "seconds": elapsed,
    }


def pr_curve(scores, corrects, n_total):
    """Precision-recall by sweeping the ZNCC score as an acceptance threshold."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    hits = np.asarray(corrects, dtype=bool)[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_total, 1)
    return (np.concatenate([[1.0], precision]),
            np.concatenate([[0.0], recall]))


def average_precision(precision, recall) -> float:
    order = np.argsort(recall)
    # numpy 2.x renamed trapz -> trapezoid and removed the old name, so the
    # lookup has to be lazy: getattr(np, "trapezoid", np.trapz) would evaluate
    # np.trapz eagerly and raise before the fallback is ever needed.
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(np.asarray(precision)[order], np.asarray(recall)[order]))


def summarize(results, tolerance) -> dict:
    errors = np.array([r["error_px"] for r in results])
    scores = [r["score"] for r in results]
    corrects = errors <= tolerance
    precision, recall = pr_curve(scores, corrects, len(results))
    return {
        "n": len(results),
        "accuracy": float(corrects.mean()) if len(results) else 0.0,
        "accuracy_5px": float((errors <= 5).mean()) if len(results) else 0.0,
        "accuracy_20px": float((errors <= 20).mean()) if len(results) else 0.0,
        "median_error_px": float(np.median(errors)) if len(results) else 0.0,
        "p90_error_px": float(np.percentile(errors, 90)) if len(results) else 0.0,
        "mean_score": float(np.mean(scores)) if results else 0.0,
        "ap": average_precision(precision, recall),
        "lattice_explained": int(sum(1 for r in results
                                     if r["lattice_explained"] and r["error_px"] > tolerance)),
        "_precision": precision.tolist(),
        "_recall": recall.tolist(),
    }


def make_plots(results, per_split, per_difficulty, out_dir: Path, tolerance: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for name, s in per_split.items():
        ax.plot(s["_recall"], s["_precision"], marker="o", markersize=2.5,
                label=f"{name} (AP={s['ap']:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"ZNCC baseline: precision-recall (tol={tolerance:g}px)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    errors = np.sort(np.array([r["error_px"] for r in results]))
    cdf = np.arange(1, len(errors) + 1) / max(len(errors), 1)
    ax.plot(errors, cdf, linewidth=2)
    ax.axvline(tolerance, color="#d62728", linestyle="--",
               label=f"tolerance {tolerance:g}px")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Localization error (px)"); ax.set_ylabel("Fraction of samples")
    ax.set_title("Error CDF"); ax.set_ylim(0, 1.02)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    names = [d for d in ("easy", "medium", "hard") if d in per_difficulty]
    accs = [per_difficulty[d]["accuracy"] for d in names]
    aps = [per_difficulty[d]["ap"] for d in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, accs, 0.4, label=f"accuracy@{tolerance:g}px")
    ax.bar(x + 0.2, aps, 0.4, label="AP")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.02); ax.set_ylabel("Score")
    ax.set_title("Baseline quality by difficulty")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = out_dir / "baseline_evaluation.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> int:
    args = parse_args()
    root = Path(args.dataset)
    sizes = args.sizes or DEFAULT_SIZES
    angles = args.angles or DEFAULT_ANGLES
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers
    if workers <= 0:
        workers = max(1, min((os.cpu_count() or 2) - 1, 6))

    jobs = []
    for split in args.splits:
        path = root / split / "annotations.json"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1
        records = json.load(open(path, "r", encoding="utf-8"))["samples"]
        if args.limit:
            records = records[:args.limit]
        for rec in records:
            jobs.append({"root": str(root), "record": rec,
                         "sizes": sizes, "angles": angles})

    print(f"ZNCC baseline on {root.resolve()}")
    print(f"  splits: {', '.join(args.splits)}   samples: {len(jobs)}   workers: {workers}")
    print(f"  sweeping {len(sizes)} template sizes x {len(angles)} rotations "
          f"= {len(sizes) * len(angles)} correlations per sample")
    print(f"  NOTE: ZNCC is a classical matcher -- there is no model and no "
          f"training step.\n")

    started = time.time()
    results = []
    if workers == 1:
        for n, job in enumerate(jobs, 1):
            results.append(evaluate_one(job))
            _progress(n, len(jobs), started)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(evaluate_one, job) for job in jobs]
            for n, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                _progress(n, len(jobs), started)
    print()

    overall = summarize(results, args.tolerance_px)
    per_split, per_difficulty = {}, {}
    by_split = defaultdict(list)
    by_diff = defaultdict(list)
    for r in results:
        by_split[r["split"]].append(r)
        by_diff[r["difficulty"]].append(r)
    for k, v in by_split.items():
        per_split[k] = summarize(v, args.tolerance_px)
    for k, v in by_diff.items():
        per_difficulty[k] = summarize(v, args.tolerance_px)

    print("=" * 72)
    print(f"ZNCC BASELINE RESULTS   (tolerance {args.tolerance_px:g} px)")
    print("=" * 72)
    header = f"{'':<12}{'n':>6}{'acc':>9}{'acc@5':>9}{'acc@20':>9}{'median':>10}{'p90':>10}{'AP':>8}"
    print(header)
    print("-" * 72)
    for name in args.splits:
        if name not in per_split:
            continue
        s = per_split[name]
        print(f"{name:<12}{s['n']:>6}{s['accuracy']:>9.3f}{s['accuracy_5px']:>9.3f}"
              f"{s['accuracy_20px']:>9.3f}{s['median_error_px']:>10.2f}"
              f"{s['p90_error_px']:>10.2f}{s['ap']:>8.3f}")
    print("-" * 72)
    for name in ("easy", "medium", "hard"):
        if name not in per_difficulty:
            continue
        s = per_difficulty[name]
        print(f"{name:<12}{s['n']:>6}{s['accuracy']:>9.3f}{s['accuracy_5px']:>9.3f}"
              f"{s['accuracy_20px']:>9.3f}{s['median_error_px']:>10.2f}"
              f"{s['p90_error_px']:>10.2f}{s['ap']:>8.3f}")
    print("-" * 72)
    s = overall
    print(f"{'OVERALL':<12}{s['n']:>6}{s['accuracy']:>9.3f}{s['accuracy_5px']:>9.3f}"
          f"{s['accuracy_20px']:>9.3f}{s['median_error_px']:>10.2f}"
          f"{s['p90_error_px']:>10.2f}{s['ap']:>8.3f}")
    print("=" * 72)
    print(f"Mean ZNCC score:        {s['mean_score']:.3f}")
    print(f"Misses explained by an integer lattice step: {s['lattice_explained']} "
          f"of {sum(1 for r in results if r['error_px'] > args.tolerance_px)} misses")
    print(f"Elapsed: {(time.time() - started) / 60:.1f} min "
          f"({(time.time() - started) / max(len(jobs), 1):.2f} s/sample)")

    payload = {
        "dataset": str(root.resolve()),
        "splits": args.splits,
        "tolerance_px": args.tolerance_px,
        "template_sizes": sizes,
        "rotations_deg": angles,
        "matcher": "ZNCC (cv2.TM_CCOEFF_NORMED), classical -- no training",
        "overall": {k: v for k, v in overall.items() if not k.startswith("_")},
        "per_split": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                      for k, v in per_split.items()},
        "per_difficulty": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                           for k, v in per_difficulty.items()},
        "samples": results,
    }
    with open(out_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out_dir / 'baseline_results.json'}")

    if not args.no_plots:
        make_plots(results, per_split, per_difficulty, out_dir, args.tolerance_px)
    return 0


def _progress(n: int, total: int, started: float) -> None:
    if n % 10 and n != total:
        return
    rate = n / max(time.time() - started, 1e-6)
    print(f"\r  {n}/{total} evaluated ({rate:.2f}/s, eta "
          f"{(total - n) / max(rate, 1e-9) / 60:.1f} min)", end="", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

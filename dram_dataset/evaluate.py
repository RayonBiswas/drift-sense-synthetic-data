"""
evaluate.py
============
Drift-Sense (Applied Materials PS-2) — measurable success-rate harness.

Runs the localization pipeline across many randomly generated test cases and
reports, per the FAQ / "Expected Solution" slide requirements:

  1. Computation time of the algorithm on a single 1000x1000 image pair.
  2. Percentage of >=30 randomized test cases landing within a stated pixel
     tolerance of the true (downsampled) location.
  3. At least one honest failure example (periodic-array confusion) with an
     explanation.

Usage
-----
    python evaluate.py --n 30 --tolerance 3 --styles dram finfet
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_generator import generate_pair, to_rgb_variant
from localize import localize


def run_eval(n: int, tolerance: float, styles, seed0: int = 1000,
             save_failure_dir: str | None = None, rgb: bool = False):
    rng = np.random.default_rng(seed0)
    records = []
    times = []
    for i in range(n):
        style = styles[i % len(styles)]
        seed = seed0 + i
        ref, wide, gt = generate_pair(style=style, seed=seed)

        if rgb:
            local_rng = np.random.default_rng(seed * 3 + 7)
            ref_in = to_rgb_variant(ref, local_rng)
            wide_in = to_rgb_variant(wide, local_rng)
        else:
            ref_in, wide_in = ref, wide

        t0 = time.perf_counter()
        result = localize(ref_in, wide_in)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        err = float(np.hypot(result.center_x - gt.center_x, result.center_y - gt.center_y))
        success = err <= tolerance
        records.append({
            "index": i,
            "style": style,
            "seed": seed,
            "gt_x": gt.center_x,
            "gt_y": gt.center_y,
            "pred_x": result.center_x,
            "pred_y": result.center_y,
            "error_px": err,
            "success": success,
            "n_candidates": result.n_candidates,
            "score": result.score,
            "time_sec": elapsed,
        })

        if save_failure_dir and not success:
            outdir = Path(save_failure_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ref if not rgb else ref_in).save(outdir / f"fail_{i}_{style}_reference.png")
            Image.fromarray(wide if not rgb else wide_in).save(outdir / f"fail_{i}_{style}_search.png")

    success_rate = 100.0 * sum(r["success"] for r in records) / len(records)
    errs = [r["error_px"] for r in records]
    summary = {
        "n": n,
        "tolerance_px": tolerance,
        "success_rate_pct": success_rate,
        "mean_error_px": statistics.mean(errs),
        "median_error_px": statistics.median(errs),
        "max_error_px": max(errs),
        "mean_time_ms": statistics.mean(times) * 1000,
        "p95_time_ms": sorted(times)[int(0.95 * (len(times) - 1))] * 1000,
        "by_style": {},
    }
    for style in set(r["style"] for r in records):
        style_recs = [r for r in records if r["style"] == style]
        summary["by_style"][style] = {
            "n": len(style_recs),
            "success_rate_pct": 100.0 * sum(r["success"] for r in style_recs) / len(style_recs),
            "mean_error_px": statistics.mean(r["error_px"] for r in style_recs),
        }
    return summary, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tolerance", type=float, default=3.0,
                     help="success tolerance in wide-search pixels")
    ap.add_argument("--styles", nargs="+", default=["dram", "finfet"])
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--rgb", action="store_true", help="run the bonus RGB path")
    ap.add_argument("--save-failures", default=None, help="dir to dump failure-case images")
    ap.add_argument("--out", default="../outputs/eval_report.json")
    args = ap.parse_args()

    summary, records = run_eval(
        args.n, args.tolerance, args.styles, args.seed0,
        save_failure_dir=args.save_failures, rgb=args.rgb,
    )

    print(json.dumps(summary, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    
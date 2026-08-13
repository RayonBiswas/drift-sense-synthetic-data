#!/usr/bin/env python3
"""
Generate the DRAM-style synthetic dataset for reference-to-search localization.

    # quick smoke test (30 pairs, a minute or so)
    python generate_dataset.py --num-samples 30 --output debug_dataset --seed 42

    # the full dataset (1000 pairs)
    python generate_dataset.py --num-samples 1000 --output dataset --seed 42

Samples are independent, so generation is embarrassingly parallel: `--workers`
fans the work across processes. `--supersample` trades fidelity for speed and
memory (each worker holds a 1000*S square canvas; S=10 is ~100 MB, S=6 is
~36 MB).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from dram_synth import params as P
from dram_synth.qc import run_all_checks
from dram_synth.sample import render_one_sample

SPLIT_RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=1000,
                   help="total image pairs across all splits (default: 1000)")
    p.add_argument("--output", default="dataset", help="output root directory")
    p.add_argument("--seed", type=int, default=42, help="master random seed")
    p.add_argument("--workers", type=int, default=0,
                   help="worker processes; 0 = auto (cpu_count-1, capped at 6), 1 = serial")
    p.add_argument("--supersample", type=int, default=10,
                   help="fine-canvas supersampling factor S; the search image is a "
                        "1000*S canvas reduced to 1000x1000 (default: 10)")
    p.add_argument("--skip-qc", action="store_true", help="skip quality-control checks")
    p.add_argument("--qc-gt-limit", type=int, default=0,
                   help="cap how many samples get the ground-truth correlation check "
                        "(0 = every sample)")
    return p.parse_args()


def split_sizes(total: int) -> dict:
    """Allocate `total` across the splits, giving any remainder to train."""
    sizes = {s: int(total * r) for s, r in SPLIT_RATIOS.items()}
    sizes["train"] += total - sum(sizes.values())
    return sizes


def build_jobs(root: Path, sizes: dict, master_seed: int, supersample: int) -> list:
    """One job per sample. Image ids run continuously across the whole dataset,
    so no id or filename can collide between splits."""
    jobs, running = [], 0
    for split in ("train", "validation", "test"):
        n = sizes[split]
        difficulties = P.build_difficulty_plan(n, master_seed, split)
        for i in range(n):
            running += 1
            jobs.append({
                "split": split,
                "index": i,
                "image_id": f"sample_{running:06d}",
                "sample_seed": P.make_sample_seed(master_seed, split, i),
                "difficulty": difficulties[i],
                "supersample": supersample,
                "reference_dir": str(root / split / "references"),
                "search_dir": str(root / split / "searches"),
            })
    return jobs


def make_dirs(root: Path, sizes: dict) -> None:
    for split in sizes:
        (root / split / "references").mkdir(parents=True, exist_ok=True)
        (root / split / "searches").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)


def write_config(root: Path, args, sizes: dict, elapsed: float) -> dict:
    """The global generator configuration -- everything needed to reproduce the
    dataset byte for byte."""
    config = {
        "generator": "dram_synth",
        "version": "1.0.0",
        "architecture": "DRAM (folded-bitline cell array)",
        "master_seed": args.seed,
        "num_samples": args.num_samples,
        "split_sizes": sizes,
        "split_ratios": SPLIT_RATIOS,
        "split_seed_offsets": P.SPLIT_SEED_OFFSET,
        "search_image_size_px": [P.SEARCH_SIZE_PX, P.SEARCH_SIZE_PX],
        "reference_image_size_range_px": list(P.REFERENCE_SIZE_RANGE),
        "nominal_footprint_px": P.NOMINAL_FOOTPRINT_PX,
        "supersample_factor": args.supersample,
        "magnification_ratio_canvas_to_search": args.supersample,
        "image_format": "PNG, 8-bit grayscale",
        "difficulty_mix": P.DIFFICULTY_MIX,
        "parameter_ranges": {
            "architecture_L1": P.ARCH_RANGES,
            "geometry_L2": P.GEOM_RANGES,
            "defects_L2": P.DEFECT_RANGES,
            "imaging_sem_scan_L3_L4_P19_P20": P.TIERS,
            "vibration_freq_cycles_per_frame": P.VIBRATION_FREQ_RANGE,
        },
        "generation_seconds": round(elapsed, 2),
        "reproduce_command": (
            f"python generate_dataset.py --num-samples {args.num_samples} "
            f"--output {args.output} --seed {args.seed} --supersample {args.supersample}"
        ),
    }
    with open(root / "metadata" / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config


def write_annotations(root: Path, split: str, records: list, args) -> None:
    payload = {
        "split": split,
        "count": len(records),
        "master_seed": args.seed,
        "split_seed_offset": P.SPLIT_SEED_OFFSET[split],
        "search_image_size": [P.SEARCH_SIZE_PX, P.SEARCH_SIZE_PX],
        "coordinate_frame": (
            "All coordinates are pixels of the final 1000x1000 search image, "
            "origin at the top-left, x to the right, y downward. bbox is the "
            "axis-aligned bounding box of the rotated reference window; quad "
            "gives that window's four exact corners."
        ),
        "samples": records,
    }
    with open(root / split / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def summarize(all_records: list, sizes: dict, config: dict, qc: dict, root: Path) -> None:
    search_noise = np.array([r["search_imaging"]["noise_sigma"] for r in all_records])
    ref_noise = np.array([r["reference_imaging"]["noise_sigma"] for r in all_records])
    search_blur = np.array([r["search_imaging"]["blur_sigma"] for r in all_records])
    ref_blur = np.array([r["reference_imaging"]["blur_sigma"] for r in all_records])
    scale = np.array([r["P09_scale"] for r in all_records])
    rotation = np.array([r["P10_rotation_deg"] for r in all_records])
    drift = np.array([r["search_imaging"]["thermal_drift_px"] for r in all_records])
    vib = np.array([r["search_imaging"]["vibration_amp_px"] for r in all_records])
    diffs = {d: sum(1 for r in all_records if r["difficulty"] == d) for d in P.DIFFICULTIES}
    with_defects = sum(1 for r in all_records if r["architecture"]["P08_defect_density"] > 0)

    print()
    print("=" * 62)
    print("DATASET SUMMARY")
    print("=" * 62)
    print(f"Total samples:        {len(all_records)}")
    print(f"Training:             {sizes['train']}")
    print(f"Validation:           {sizes['validation']}")
    print(f"Test:                 {sizes['test']}")
    print(f"Image size:           {P.SEARCH_SIZE_PX} x {P.SEARCH_SIZE_PX}")
    print(f"Architecture:         DRAM")
    print(f"Average noise level:  {search_noise.mean():.2f} sigma (search), "
          f"{ref_noise.mean():.2f} sigma (reference)")
    print(f"Average blur:         {search_blur.mean():.2f} px (search), "
          f"{ref_blur.mean():.2f} px (reference)")
    print(f"Average scale:        {scale.mean():.3f}x  [{scale.min():.2f} - {scale.max():.2f}]")
    print(f"Average rotation:     {np.abs(rotation).mean():.2f} deg mean absolute  "
          f"[{rotation.min():+.2f} - {rotation.max():+.2f}]")
    print(f"Average thermal drift:{drift.mean():6.2f} px per frame (search)")
    print(f"Average vibration:    {vib.mean():.2f} px amplitude (search)")
    print(f"Difficulty mix:       easy {diffs['easy']}  medium {diffs['medium']}  "
          f"hard {diffs['hard']}")
    print(f"Samples with defects: {with_defects} ({100.0 * with_defects / len(all_records):.0f}%)")
    if qc:
        print(f"Transform maths:      {qc['transform_math_error_px']:.2f} px worst case "
              f"(analytic test, non-periodic specimen)")
        print(f"Ground-truth error:   median {qc['gt_offset_median_px']:.2f} px, "
              f"p95 {qc['gt_offset_p95_px']:.2f} px  (verified on {qc['gt_checked']} samples, "
              f"{qc['gt_lattice_aliased']} lattice-aliased)")
        print(f"Noise independence:   max |corr| {qc['noise_corr_max']:.3f}")
    print(f"Stored at:            {root.resolve()}")
    print("=" * 62)


def main() -> int:
    args = parse_args()
    root = Path(args.output)
    sizes = split_sizes(args.num_samples)

    workers = args.workers
    if workers <= 0:
        workers = max(1, min((os.cpu_count() or 2) - 1, 6))

    print(f"Generating {args.num_samples} pairs into {root.resolve()}")
    print(f"  splits: train {sizes['train']} / validation {sizes['validation']} "
          f"/ test {sizes['test']}")
    print(f"  master seed {args.seed}, supersample {args.supersample}x, {workers} worker(s)")

    make_dirs(root, sizes)
    jobs = build_jobs(root, sizes, args.seed, args.supersample)

    started = time.time()
    results = {}

    if workers == 1:
        for n, job in enumerate(jobs, 1):
            results[job["image_id"]] = render_one_sample(job)
            _progress(n, len(jobs), started)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(render_one_sample, job): job for job in jobs}
            for n, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    results[job["image_id"]] = fut.result()
                except Exception as exc:
                    raise RuntimeError(f"sample {job['image_id']} failed: {exc}") from exc
                _progress(n, len(jobs), started)

    elapsed = time.time() - started
    print(f"\n  rendered {len(jobs)} pairs in {elapsed / 60:.1f} min "
          f"({elapsed / max(len(jobs), 1):.2f} s/pair)")

    ordered = [results[j["image_id"]] for j in jobs]
    for rec in ordered:
        rec.pop("_stats", None)
    for split in sizes:
        write_annotations(root, split, [r for r in ordered if r["split"] == split], args)
    config = write_config(root, args, sizes, elapsed)
    print(f"  wrote annotations.json x3 and metadata/generation_config.json")

    qc = None
    if not args.skip_qc:
        print("\nRunning quality-control checks:")
        qc = run_all_checks(root, args.num_samples, sizes, args.seed,
                            gt_sample_limit=args.qc_gt_limit)
        print("  all 12 checks passed")

    summarize(ordered, sizes, config, qc, root)
    return 0


def _progress(n: int, total: int, started: float) -> None:
    if n % 10 and n != total:
        return
    rate = n / max(time.time() - started, 1e-6)
    eta = (total - n) / max(rate, 1e-9)
    print(f"\r  {n}/{total} pairs  ({rate:.2f}/s, eta {eta / 60:.1f} min)",
          end="", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
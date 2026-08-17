#!/usr/bin/env python3
"""
Step 0 gate: turn the raw ZNCC baseline results into a step report.

    python scripts/step0_report.py

Reads  reports/step0_baseline_zncc/baseline_results.json
Writes reports/step0_baseline_zncc/report.json

Step 0 has no accuracy threshold to clear -- its job is to *establish* the
number that every later step is measured against. It passes if all 200 held-out
pairs were scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "step0_baseline_zncc"
MIN_SAMPLES = 200


def lattice_explained(dx, dy, arch, max_steps=100, residual_px=3.0) -> bool:
    """Does the error resolve to a whole number of lattice repeats?

    The baseline script caps this at 6 steps, which structurally cannot catch a
    200 px error (~16 bit-line pitches), so the cap is lifted here.
    """
    px, py = arch["bit_line_pitch"], arch["P01_word_line_pitch"]
    if px <= 0 or py <= 0:
        return False
    nx, ny = round(dx / px), round(dy / py)
    if abs(nx) > max_steps or abs(ny) > max_steps:
        return False
    return bool(np.hypot(dx - nx * px, dy - ny * py) <= residual_px)


def main() -> int:
    raw_path = OUT_DIR / "baseline_results.json"
    if not raw_path.exists():
        print(f"missing {raw_path} -- run evaluate_dataset.py first")
        return 1
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    samples = raw["samples"]

    arch = {}
    for split in ("validation", "test"):
        path = ROOT / "dataset" / split / "annotations.json"
        for rec in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            arch[rec["image_id"]] = rec["architecture"]

    err = np.array([s["error_px"] for s in samples])
    sec = np.array([s["seconds"] for s in samples])
    score = np.array([s["score"] for s in samples])
    miss = err > 10.0

    n_lattice = sum(
        1 for s in samples
        if s["error_px"] > 10.0
        and lattice_explained(s["dx"], s["dy"], arch[s["image_id"]])
    )

    report = {
        "step": 0,
        "name": "baseline_zncc",
        "gate": "all 200 held-out pairs scored; no accuracy threshold",
        "status": "PASS" if len(samples) >= MIN_SAMPLES else "FAIL",
        "headline": f"n={len(samples)} acc@5px={100 * (err <= 5).mean():.1f}% "
                    f"median={np.median(err):.1f}px {sec.mean():.2f}s/pair",
        "metrics": {
            "n": len(samples),
            "accuracy_pct": {
                f"@{t}px": round(100 * float((err <= t).mean()), 1)
                for t in (1, 3, 5, 10, 20, 50)
            },
            "median_error_px": round(float(np.median(err)), 2),
            "p90_error_px": round(float(np.percentile(err, 90)), 2),
            "max_error_px": round(float(err.max()), 2),
            "seconds_per_pair": {
                "mean": round(float(sec.mean()), 3),
                "median": round(float(np.median(sec)), 3),
                "p95": round(float(np.percentile(sec, 95)), 3),
                "wall_clock_6_workers": round(float(sec.sum() / 6 / len(sec)), 3),
            },
            "mean_zncc_score": {
                "hits": round(float(score[~miss].mean()), 3),
                "misses": round(float(score[miss].mean()), 3),
            },
            "failure_anatomy": {
                "n_misses": int(miss.sum()),
                "near_10_30px": int(((err > 10) & (err <= 30)).sum()),
                "mid_30_150px": int(((err > 30) & (err <= 150)).sum()),
                "gross_over_150px": int((err > 150).sum()),
                "lattice_explained": n_lattice,
                "lattice_explained_pct": round(100 * n_lattice / max(int(miss.sum()), 1), 1),
                "lattice_chance_level_pct": 23.0,
            },
            "per_difficulty_accuracy_at_10px": {
                k: round(100 * v["accuracy"], 1)
                for k, v in raw["per_difficulty"].items()
            },
        },
    }

    path = OUT_DIR / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Measure v1 against v2 and against v2 with individual stages switched off, on the
same held-out pairs, so each stage's contribution is a number rather than a
claim.

    python scripts/ablate_localizers.py --dataset dataset_drift --split test

Every variant sees identical inputs and is scored identically: Euclidean
distance from the predicted centre to the annotated centre, in search pixels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dram_dataset"))

import cv2  # noqa: E402


def variants():
    from localize_v2 import Config
    return {
        "v1 (baseline)": None,
        "v2 (bp+sub+ctr)": Config(envelope="off"),
        "v2 -bandpass": Config(envelope="off", bandpass=False),
        "v2 -subpixel": Config(envelope="off", subpixel=False),
        "v2 -centre": Config(envelope="off", center_prior=False),
        "v2 +env rerank": Config(envelope="rerank"),
        "v2 +env add": Config(envelope="add"),
    }


def run_one(job):
    from localize import localize
    from localize_v2 import localize_v2

    ref = cv2.imread(job["ref"], cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(job["search"], cv2.IMREAD_GRAYSCALE)
    gx, gy = job["gx"], job["gy"]
    out = {"difficulty": job["difficulty"], "mode": job["mode"]}

    for name, cfg in variants().items():
        t = time.time()
        if cfg is None:
            r = localize(ref, search)
            cx, cy, conf = r.center_x, r.center_y, float(r.n_candidates)
        else:
            r = localize_v2(ref, search, cfg)
            cx, cy, conf = r.center_x, r.center_y, float(r.n_candidates)
        out[name] = {
            "err": float(np.hypot(cx - gx, cy - gy)),
            "conf": conf,
            "psr": float(getattr(r, "psr", 0.0)),
            "env": float(getattr(r, "envelope_agreement", 0.0)),
            "sec": time.time() - t,
        }
    return out


def summarize(results, name):
    err = np.array([r[name]["err"] for r in results])
    sec = np.array([r[name]["sec"] for r in results])
    return {
        "n": int(err.size),
        "acc@1px": round(100 * float((err <= 1).mean()), 1),
        "acc@5px": round(100 * float((err <= 5).mean()), 1),
        "acc@10px": round(100 * float((err <= 10).mean()), 1),
        "acc@50px": round(100 * float((err <= 50).mean()), 1),
        "median_err_px": round(float(np.median(err)), 1),
        "p90_err_px": round(float(np.percentile(err, 90)), 1),
        "sec_per_pair": round(float(sec.mean()), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset_drift")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="reports/localizer_ablation/report.json")
    args = ap.parse_args()

    root = os.path.join(ROOT, args.dataset, args.split)
    payload = json.load(open(os.path.join(root, "annotations.json"), encoding="utf-8"))
    recs = [v for v in payload.values()
            if isinstance(v, list) and v and isinstance(v[0], dict)][0]
    if args.limit:
        recs = recs[:args.limit]

    jobs = [{
        "ref": os.path.join(root, r["reference_filename"]),
        "search": os.path.join(root, r["search_filename"]),
        "gx": r["center"]["x"], "gy": r["center"]["y"],
        "difficulty": r["difficulty"], "mode": r.get("position_mode", "?"),
    } for r in recs]

    print(f"scoring {len(jobs)} pairs from {args.dataset}/{args.split} "
          f"across {len(variants())} variants on {args.workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_one, jobs))
    print(f"done in {time.time() - t0:.0f}s\n")

    names = list(variants())
    table = {n: summarize(results, n) for n in names}

    hdr = f"{'variant':18s} {'acc@1px':>8s} {'acc@5px':>8s} {'acc@50px':>9s} {'median':>9s} {'p90':>8s} {'s/pair':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for n in names:
        t = table[n]
        print(f"{n:18s} {t['acc@1px']:7.1f}% {t['acc@5px']:7.1f}% {t['acc@50px']:8.1f}% "
              f"{t['median_err_px']:8.1f} {t['p90_err_px']:7.1f} {t['sec_per_pair']:7.2f}")

    # Which signal best answers "should I trust this answer?" A localizer that
    # knows when it is lost is far more useful than one that is silently wrong,
    # because the tool can re-image instead of writing to the wrong site.
    print("\nCONFIDENCE GATING -- can the method tell when it is right?")
    print(f"{'signal':26s} {'threshold':>10s} {'kept':>6s} {'acc@5px kept':>13s} "
          f"{'acc@5px dropped':>16s} {'median kept':>12s}")
    print("-" * 90)
    err_v1 = np.array([r["v1 (baseline)"]["err"] for r in results])
    riv_v1 = np.array([r["v1 (baseline)"]["conf"] for r in results])
    err_v2 = np.array([r["v2 (bp+sub+ctr)"]["err"] for r in results])
    psr_v2 = np.array([r["v2 (bp+sub+ctr)"]["psr"] for r in results])
    env_v2 = np.array([r["v2 (bp+sub+ctr)"]["env"] for r in results])

    gate_report = {}
    riv_v2 = np.array([r["v2 (bp+sub+ctr)"]["conf"] for r in results])
    trials = [("v1 rival count", riv_v1, err_v1, [(4, "<= 4"), (2, "<= 2"), (1, "== 1")], True),
              ("v2 rival count", riv_v2, err_v2, [(4, "<= 4"), (2, "<= 2"), (1, "== 1")], True)]
    for thr in (2.8, 3.2, 3.6):
        trials.append((f"v2 PSR", psr_v2, err_v2, [(thr, f">= {thr}")], False))
    for thr in (0.70, 0.78, 0.85):
        trials.append((f"v2 envelope agreement", env_v2, err_v2, [(thr, f">= {thr}")], False))

    for label, sig, err, thresholds, invert in trials:
        for thr, tlabel in thresholds:
            keep = sig <= thr if invert else sig >= thr
            if keep.sum() == 0 or keep.sum() == len(keep):
                acc_k = 100 * float((err[keep] <= 5).mean()) if keep.sum() else float("nan")
                acc_d = float("nan")
            else:
                acc_k = 100 * float((err[keep] <= 5).mean())
                acc_d = 100 * float((err[~keep] <= 5).mean())
            med_k = float(np.median(err[keep])) if keep.sum() else float("nan")
            gate_report[f"{label} {tlabel}"] = {
                "kept": int(keep.sum()), "acc@5px_kept": round(acc_k, 1),
                "acc@5px_dropped": round(acc_d, 1), "median_kept": round(med_k, 1)}
            print(f"{label:26s} {tlabel:>10s} {keep.sum():5d} {acc_k:12.1f}% "
                  f"{acc_d:15.1f}% {med_k:12.1f}")

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "split": args.split,
                   "n": len(jobs), "variants": table, "gating": gate_report},
                  f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

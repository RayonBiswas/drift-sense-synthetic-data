#!/usr/bin/env python3
"""
Progress check for the Drift-Sense localization pipeline.

    python check.py            # status of every step
    python check.py 0          # one step, with its full metrics

Each step of the plan writes reports/step<N>_<name>/report.json containing its
own measurements and a PASS/FAIL. This script only *reads* those files -- a step
is PASS because its report says so, not because anyone claimed it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

# The plan. A step with no report yet shows as "not run".
STEPS = [
    (0, "baseline_zncc", "Honest ZNCC baseline on 200 held-out pairs"),
    (1, "spec_tiebreak", "Centre-closest tie-break + multi-candidate return"),
    (2, "data_loader", "Pair loader and heatmap label contract"),
    (3, "model_skeleton", "Torch env, forward/backward pass, param count"),
    (4, "overfit_32", "Overfit 32 samples to <2px -- the design gate"),
    (5, "train_full", "Full training with lattice hard-negative mining"),
    (6, "hybrid_infer", "Heatmap -> top-K -> ZNCC refine -> tie-break"),
    (7, "slide_evidence", "30+ case table, success + honest failure figures"),
]

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def load(step: int, name: str):
    path = REPORTS / f"step{step}_{name}" / "report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "FAIL", "headline": f"unreadable report.json: {exc}"}


def colour(status: str) -> str:
    if status == "PASS":
        return f"{GREEN}PASS{RESET}"
    if status == "FAIL":
        return f"{RED}FAIL{RESET}"
    return f"{DIM}----{RESET}"


def main() -> int:
    if len(sys.argv) > 1:
        want = int(sys.argv[1])
        match = [s for s in STEPS if s[0] == want]
        if not match:
            print(f"no step {want}; steps are 0-{STEPS[-1][0]}")
            return 1
        step, name, _ = match[0]
        report = load(step, name)
        if report is None:
            print(f"step {step} ({name}) has not been run")
            return 1
        print(json.dumps(report, indent=2))
        return 0 if report.get("status") == "PASS" else 1

    print(f"\n{BOLD}Drift-Sense pipeline{RESET}   {DIM}reports/{RESET}\n")
    failed = False
    for step, name, description in STEPS:
        report = load(step, name)
        status = report.get("status", "FAIL") if report else "----"
        detail = report.get("headline", "") if report else description
        if status == "FAIL":
            failed = True
        print(f"  STEP {step}  {name:<16} {colour(status)}  {detail}")
    print()
    print(f"  {DIM}python check.py <n>  for one step's full metrics{RESET}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

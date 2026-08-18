#!/usr/bin/env python3
"""
Draw one figure per evaluation axis: what the sweep looks like, and where the
localizer actually put the answer.

    python scripts/visualize_eval_suite.py --replicate 1

Each figure is a row of the six levels of one axis, all from the same replicate,
so the specimen is identical across the row and only the swept parameter moves.
Every panel shows the search image with:

    green  the annotated ground-truth quadrilateral
    red    the centre v2 returned, with its error in pixels

and the stored reference for that case inset at the top left, labelled with its
stored side length in pixels, so the magnification gap between the two captures
is visible rather than asserted. The inset is drawn at a fixed panel fraction,
not to scale -- compare its label against the ~100 px ground-truth box.

Requires reports/eval_suite/report.json, so run scripts/run_eval_suite.py first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOL = 5.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="eval_suite")
    ap.add_argument("--report", default="reports/eval_suite/report.json")
    ap.add_argument("--replicate", type=int, default=1,
                    help="which replicate to draw (the specimen is fixed within one)")
    ap.add_argument("--out", default="reports/eval_suite/figures")
    args = ap.parse_args()

    suite = ROOT / args.suite / "eval"
    ann = {r["case"]["case_id"]: r
           for r in json.load(open(suite / "annotations.json", encoding="utf-8"))["samples"]}
    report = json.load(open(ROOT / args.report, encoding="utf-8"))
    rows = [r for r in report["cases"] if r["replicate"] == args.replicate]
    if not rows:
        print(f"no cases for replicate {args.replicate}")
        return 1

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    by_axis = {}
    for r in rows:
        by_axis.setdefault(r["axis"], []).append(r)

    for axis in sorted(by_axis):
        group = by_axis[axis]
        fig, axarr = plt.subplots(1, len(group), figsize=(4.1 * len(group), 4.9))
        if len(group) == 1:
            axarr = [axarr]
        for ax, row in zip(axarr, group):
            rec = ann[row["case_id"]]
            search = cv2.imread(str(suite / rec["search_filename"]), cv2.IMREAD_GRAYSCALE)
            ref = cv2.imread(str(suite / rec["reference_filename"]), cv2.IMREAD_GRAYSCALE)
            ax.imshow(search, cmap="gray", vmin=0, vmax=255)

            ax.add_patch(Polygon(rec["quad"], closed=True, fill=False,
                                 edgecolor="#2ca02c", linewidth=2.0))
            ax.add_patch(Circle((row["gx"], row["gy"]), 4.0, color="#2ca02c"))

            hit = row["v2"]["err"] <= TOL
            colour = "#2ca02c" if hit else "#d62728"
            ax.add_patch(Circle((row["v2"]["x"], row["v2"]["y"]), 7.0, fill=False,
                                edgecolor=colour, linewidth=2.0))
            if not hit:
                ax.plot([row["gx"], row["v2"]["x"]], [row["gy"], row["v2"]["y"]],
                        color="#d62728", linewidth=1.2, linestyle="--")

            # The stored reference. Drawn at a fixed inset size and labelled with
            # its true stored side, so the reader compares the label against the
            # ~100 px ground-truth box rather than trusting the drawn size.
            inset = ax.inset_axes([0.02, 0.72, 0.26, 0.26])
            inset.imshow(ref, cmap="gray", vmin=0, vmax=255)
            inset.set_xticks([]); inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_edgecolor("#1f77b4")
                spine.set_linewidth(1.6)
            inset.set_title(f"ref {rec['reference_width']}px", fontsize=7,
                            color="#1f77b4", pad=2)

            ax.set_title(f"{row['level']}\nerr {row['v2']['err']:.2f} px, "
                         f"{row['v2']['rivals']} rivals",
                         fontsize=10, color=colour)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Axis {axis} -- {group[0]['axis_label']}   "
                     f"(replicate {args.replicate}: one specimen, one parameter moving)\n"
                     f"green = ground truth, red = v2 answer when it misses by more "
                     f"than {TOL:.0f} px", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        path = out_dir / f"axis_{axis}_r{args.replicate}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

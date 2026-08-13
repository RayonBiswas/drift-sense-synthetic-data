#!/usr/bin/env python3
"""
Visual inspection tool for the generated dataset.

    # 6 detail panels + a 25-sample contact sheet, from the train split
    python visualize_dataset.py --dataset dataset --num-panels 6

    # inspect a specific sample
    python visualize_dataset.py --dataset dataset --image-id sample_000123

Each detail panel shows the reference image, the search image, the search image
with the ground-truth box drawn on it, the zoomed ground-truth region, and the
sample's metadata. The contact sheet puts 25 random search images side by side
with their boxes, which is the fastest way to eyeball whether placement really
is spread across the frame and whether the difficulty mix looks right.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DIFFICULTY_COLOR = {"easy": "#2ca02c", "medium": "#ff7f0e", "hard": "#d62728"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset", help="dataset root directory")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--num-panels", type=int, default=6, help="detail panels to render")
    p.add_argument("--contact-sheet", type=int, default=25,
                   help="samples on the contact sheet (0 to skip)")
    p.add_argument("--image-id", default=None, help="render one specific sample")
    p.add_argument("--output", default=None,
                   help="where to write figures (default: ./visualizations, kept "
                        "outside the dataset directory so it stays exactly the "
                        "published structure)")
    p.add_argument("--seed", type=int, default=0, help="seed for choosing samples")
    return p.parse_args()


def load_annotations(root: Path, split: str) -> list:
    with open(root / split / "annotations.json", "r", encoding="utf-8") as f:
        return json.load(f)["samples"]


def read_pair(root: Path, rec: dict):
    base = root / rec["split"]
    ref = cv2.imread(str(base / rec["reference_filename"]), cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(str(base / rec["search_filename"]), cv2.IMREAD_GRAYSCALE)
    return ref, search


def metadata_text(rec: dict) -> str:
    a = rec["architecture"]
    ri, si = rec["reference_imaging"], rec["search_imaging"]
    b = rec["bbox"]
    defects = {k: v for k, v in rec["defects"].items() if v}
    return "\n".join([
        f"{rec['image_id']}   [{rec['difficulty']}]   split={rec['split']}",
        f"seed = {rec['random_seed']}",
        "",
        "GROUND TRUTH",
        f"  bbox    ({b['x1']:.1f}, {b['y1']:.1f}) - ({b['x2']:.1f}, {b['y2']:.1f})",
        f"  centre  ({rec['center']['x']:.1f}, {rec['center']['y']:.1f})",
        f"  footprint {rec['footprint_px']:.1f} px   ref stored "
        f"{rec['reference_width']}x{rec['reference_height']}",
        "",
        "ARCHITECTURE  (L1/L2)",
        f"  P01 word pitch   {a['P01_word_line_pitch']:.2f}",
        f"  P02 bit ratio    {a['P02_bit_to_word_pitch_ratio']:.2f}"
        f"  -> bit pitch {a['bit_line_pitch']:.2f}",
        f"  P03 width frac   {a['P03_line_width_frac']:.2f}",
        f"  P04 contact frac {a['P04_contact_diameter_frac']:.2f}",
        f"  P05 lines/block  {a['P05_block_line_count']}",
        f"  P06 jitter       {a['P06_spacing_jitter']:.3f}",
        f"  P08 defect dens. {a['P08_defect_density']:.3f}",
        "",
        "IMAGING            reference / search",
        f"  P09 scale        {rec['P09_scale']:.3f}",
        f"  P10 rotation     {rec['P10_rotation_deg']:+.2f} deg",
        f"  P11 blur         {ri['blur_sigma']:.2f} / {si['blur_sigma']:.2f}",
        f"  P12 brightness   {ri['brightness']:+.1f} / {si['brightness']:+.1f}",
        f"  P13 contrast     {ri['contrast']:.2f} / {si['contrast']:.2f}",
        f"  P14 edge gain    {ri['edge_gain']:.2f} / {si['edge_gain']:.2f}",
        f"  P15 shading      {ri['shading_amplitude']:.2f} / {si['shading_amplitude']:.2f}",
        f"  P16 noise sigma  {ri['noise_sigma']:.1f} / {si['noise_sigma']:.1f}",
        f"      dose         {ri['poisson_dose']:.0f} / {si['poisson_dose']:.0f}",
        f"  P19 therm. drift {ri['thermal_drift_px']:.2f} / {si['thermal_drift_px']:.2f} px",
        f"  P20 vibration    {ri['vibration_amp_px']:.2f} / {si['vibration_amp_px']:.2f} px"
        f" @ {si['vibration_freq_cycles']:.0f} cyc",
        "",
        f"DEFECTS  {defects if defects else 'none'}",
    ])


def render_panel(root: Path, rec: dict, out_path: Path) -> None:
    ref, search = read_pair(root, rec)
    b = rec["bbox"]
    colour = DIFFICULTY_COLOR.get(rec["difficulty"], "#1f77b4")

    fig = plt.figure(figsize=(19, 4.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1.15, 1.15, 1, 1.35], wspace=0.22)

    ax = fig.add_subplot(gs[0])
    ax.imshow(ref, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Reference  {ref.shape[1]}x{ref.shape[0]} px", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[1])
    ax.imshow(search, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Search  1000x1000 px", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[2])
    ax.imshow(search, cmap="gray", vmin=0, vmax=255)
    ax.add_patch(Rectangle((b["x1"], b["y1"]), b["x2"] - b["x1"], b["y2"] - b["y1"],
                           fill=False, edgecolor=colour, linewidth=1.8))
    ax.add_patch(Polygon(rec["quad"], closed=True, fill=False,
                         edgecolor="#00e5ff", linewidth=1.0, linestyle="--"))
    ax.plot(rec["center"]["x"], rec["center"]["y"], "+", color=colour, markersize=11)
    ax.set_title("Search + ground truth", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[3])
    pad = 14
    y0 = max(int(b["y1"]) - pad, 0); y1 = min(int(b["y2"]) + pad, search.shape[0])
    x0 = max(int(b["x1"]) - pad, 0); x1 = min(int(b["x2"]) + pad, search.shape[1])
    ax.imshow(search[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
    ax.add_patch(Rectangle((b["x1"] - x0, b["y1"] - y0), b["x2"] - b["x1"],
                           b["y2"] - b["y1"], fill=False, edgecolor=colour, linewidth=1.8))
    ax.set_title("Zoomed ground-truth region", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[4])
    ax.axis("off")
    ax.text(0.0, 1.0, metadata_text(rec), va="top", ha="left",
            fontsize=6.6, family="monospace", transform=ax.transAxes)

    fig.suptitle(f"{rec['image_id']}  -  difficulty: {rec['difficulty']}",
                 fontsize=12, color=colour, y=0.99)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_contact_sheet(root: Path, records: list, out_path: Path, n: int = 25) -> None:
    picked = records[:n]
    side = int(len(picked) ** 0.5 + 0.999) or 1
    fig, axes = plt.subplots(side, side, figsize=(2.5 * side, 2.62 * side))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, rec in zip(axes, picked):
        _, search = read_pair(root, rec)
        b = rec["bbox"]
        colour = DIFFICULTY_COLOR.get(rec["difficulty"], "#1f77b4")
        ax.imshow(search, cmap="gray", vmin=0, vmax=255)
        ax.add_patch(Rectangle((b["x1"], b["y1"]), b["x2"] - b["x1"], b["y2"] - b["y1"],
                               fill=False, edgecolor=colour, linewidth=1.4))
        ax.set_title(f"{rec['image_id'].replace('sample_', '#')}  {rec['difficulty']}",
                     fontsize=7.5, color=colour, pad=2)
        ax.axis("off")
    for ax in axes[len(picked):]:
        ax.axis("off")

    fig.suptitle(f"Contact sheet - {len(picked)} random samples with ground-truth boxes",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=95, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(args.dataset)
    out_dir = Path(args.output) if args.output else Path(__file__).resolve().parent / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image_id:
        for split in ("train", "validation", "test"):
            if not (root / split / "annotations.json").exists():
                continue
            for rec in load_annotations(root, split):
                if rec["image_id"] == args.image_id:
                    path = out_dir / f"panel_{rec['image_id']}.png"
                    render_panel(root, rec, path)
                    print(f"wrote {path}")
                    return 0
        print(f"sample '{args.image_id}' not found", file=sys.stderr)
        return 1

    records = load_annotations(root, args.split)
    rng = random.Random(args.seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    for rec in shuffled[:args.num_panels]:
        path = out_dir / f"panel_{rec['image_id']}.png"
        render_panel(root, rec, path)
        print(f"wrote {path}")

    if args.contact_sheet > 0:
        path = out_dir / f"contact_sheet_{args.split}.png"
        render_contact_sheet(root, shuffled, path, n=args.contact_sheet)
        print(f"wrote {path}")

    print(f"\nfigures in {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
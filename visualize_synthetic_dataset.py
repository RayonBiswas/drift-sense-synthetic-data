#!/usr/bin/env python3
"""
Visualization script for synthetic dataset inspection.

Creates contact sheets and detailed visualizations of random samples.

Usage:
    python visualize_synthetic_dataset.py --dataset dataset --output visualizations --num-samples 25
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_annotations(dataset_dir: Path) -> List[dict]:
    """Load annotations from dataset."""
    ann_file = dataset_dir / "annotations.json"
    if not ann_file.exists():
        logger.error(f"Annotations file not found: {ann_file}")
        return []
    
    with open(ann_file, "r") as f:
        return json.load(f)


def create_contact_sheet(
    dataset_dir: Path,
    annotations: List[dict],
    num_samples: int = 25,
    output_path: Path = None,
    seed: int = 42,
) -> None:
    """Create a contact sheet with random samples."""
    if output_path is None:
        output_path = Path("contact_sheet.png")
    
    logger.info(f"Creating contact sheet with {num_samples} samples...")
    
    # Randomly select samples
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(annotations), min(num_samples, len(annotations)), replace=False)
    selected_ann = [annotations[i] for i in selected_indices]
    
    # Create grid (5x5 for 25 samples)
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), dpi=100)
    axes = np.atleast_2d(axes).flatten()
    
    for idx, ann in enumerate(selected_ann):
        ax = axes[idx]
        
        # Load images
        search_path = dataset_dir / "searches" / ann["search_filename"]
        ref_path = dataset_dir / "references" / ann["reference_filename"]
        
        if not search_path.exists() or not ref_path.exists():
            logger.warning(f"Images not found for {ann['sample_id']}")
            ax.text(0.5, 0.5, "Images\nNot Found", ha="center", va="center", transform=ax.transAxes)
            continue
        
        search_img = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
        
        # Display search image
        ax.imshow(search_img, cmap="gray")
        
        # Draw bounding box
        bbox = ann["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        
        # Draw center
        cx, cy = ann["center"]["x"], ann["center"]["y"]
        ax.plot(cx, cy, "r+", markersize=15, markeredgewidth=2)
        
        # Title
        difficulty = ann.get("difficulty_level", "?")
        noise = ann.get("gauss_sigma_search", 0)
        title = f"{ann['sample_id']} ({difficulty})\nNoise σ={noise:.1f}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(len(selected_ann), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    logger.info(f"Contact sheet saved to {output_path}")
    plt.close()


def create_sample_detail_sheet(
    dataset_dir: Path,
    annotation: dict,
    output_path: Path = None,
) -> None:
    """Create a detailed visualization of a single sample."""
    if output_path is None:
        output_path = Path(f"sample_detail_{annotation['sample_id']}.png")
    
    logger.info(f"Creating detail sheet for {annotation['sample_id']}...")
    
    # Load images
    search_path = dataset_dir / "searches" / annotation["search_filename"]
    ref_path = dataset_dir / "references" / annotation["reference_filename"]
    
    if not search_path.exists() or not ref_path.exists():
        logger.error(f"Images not found for {annotation['sample_id']}")
        return
    
    search_img = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10), dpi=100)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Full search image with bbox
    ax_search = fig.add_subplot(gs[0:2, 0:2])
    ax_search.imshow(search_img, cmap="gray")
    
    bbox = annotation["bbox"]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none")
    ax_search.add_patch(rect)
    
    cx, cy = annotation["center"]["x"], annotation["center"]["y"]
    ax_search.plot(cx, cy, "r+", markersize=20, markeredgewidth=2)
    ax_search.set_title("Search Image with Ground Truth", fontsize=12, fontweight="bold")
    ax_search.set_xticks([])
    ax_search.set_yticks([])
    
    # Zoomed ground truth region
    ax_zoom = fig.add_subplot(gs[0:2, 2])
    zoom_size = 150
    crop_x1 = max(0, int(cx - zoom_size // 2))
    crop_x2 = min(1000, crop_x1 + zoom_size)
    crop_y1 = max(0, int(cy - zoom_size // 2))
    crop_y2 = min(1000, crop_y1 + zoom_size)
    
    zoomed = search_img[crop_y1:crop_y2, crop_x1:crop_x2]
    ax_zoom.imshow(zoomed, cmap="gray")
    ax_zoom.set_title("Zoomed Region", fontsize=12, fontweight="bold")
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    
    # Reference image
    ax_ref = fig.add_subplot(gs[2, 0])
    ax_ref.imshow(ref_img, cmap="gray")
    ax_ref.set_title(f"Reference ({annotation['reference_width']}×{annotation['reference_height']})", fontsize=10)
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])
    
    # Metadata text
    ax_meta = fig.add_subplot(gs[2, 1:3])
    ax_meta.axis("off")
    
    metadata_text = f"""
METADATA:
Split: {annotation['split']}  |  Difficulty: {annotation['difficulty_level']}  |  Seed: {annotation['random_seed']}

ARCHITECTURE:
Pitch: {annotation['pitch_px']:.1f}px  |  Line Width: {annotation['line_width_px']:.1f}px  |  Contacts: {annotation['contact_diameter_px']:.1f}px
Lines: {annotation['num_h_lines']}H × {annotation['num_v_lines']}V

TRANSFORMATIONS:
Rotation: {annotation['rotation_deg']:.2f}°  |  Scale: {annotation['scale_factor']:.3f}  |  Blur σ: {annotation['blur_sigma']:.2f}
Contrast: {annotation['contrast']:.2f}  |  Brightness: {annotation['brightness']:+d}

NOISE & EFFECTS:
Gauss (ref): σ={annotation['gauss_sigma_ref']:.2f}  |  Gauss (search): σ={annotation['gauss_sigma_search']:.2f}
Poisson: {annotation['poisson_intensity']:.3f}  |  Edge Brightening: {annotation['edge_brightening_strength']:.2f}

PHYSICAL ERRORS:
Thermal Drift: {annotation['thermal_drift_px']:.2f}px  |  Thermal Intensity: {annotation['thermal_intensity_variation_pct']:.1f}%
Acoustic Jitter: {annotation['acoustic_jitter_amplitude_px']:.2f}px @ {annotation['acoustic_jitter_frequency_hz']:.1f}Hz
Vibrational Drift: ({annotation['vibrational_drift_dx_px']:.2f}, {annotation['vibrational_drift_dy_px']:.2f})px  |  Creep: {annotation['vibrational_creep_rate']:.2f}

GROUND TRUTH:
BBox: ({bbox['x1']}, {bbox['y1']}) → ({bbox['x2']}, {bbox['y2']})  |  Center: ({cx:.1f}, {cy:.1f})
    """
    
    ax_meta.text(0.05, 0.95, metadata_text, transform=ax_meta.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    logger.info(f"Detail sheet saved to {output_path}")
    plt.close()


def create_difficulty_distribution_plot(
    annotations: List[dict],
    output_path: Path = None,
) -> None:
    """Create plots showing dataset statistics."""
    if output_path is None:
        output_path = Path("dataset_statistics.png")
    
    logger.info("Creating dataset statistics plots...")
    
    # Extract statistics
    difficulties = [ann.get("difficulty_level", "unknown") for ann in annotations]
    noise_ref = [ann.get("gauss_sigma_ref", 0) for ann in annotations]
    noise_search = [ann.get("gauss_sigma_search", 0) for ann in annotations]
    blur = [ann.get("blur_sigma", 0) for ann in annotations]
    rotation = [abs(ann.get("rotation_deg", 0)) for ann in annotations]
    scale = [ann.get("scale_factor", 1) for ann in annotations]
    contrast = [ann.get("contrast", 1) for ann in annotations]
    thermal = [ann.get("thermal_drift_px", 0) for ann in annotations]
    acoustic = [ann.get("acoustic_jitter_amplitude_px", 0) for ann in annotations]
    
    splits = [ann.get("split", "unknown") for ann in annotations]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=100)
    
    # Difficulty distribution
    ax = axes[0, 0]
    difficulty_counts = {}
    for d in difficulties:
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
    ax.bar(difficulty_counts.keys(), difficulty_counts.values(), color=["green", "orange", "red"])
    ax.set_title("Difficulty Distribution")
    ax.set_ylabel("Count")
    
    # Split distribution
    ax = axes[0, 1]
    split_counts = {}
    for s in splits:
        split_counts[s] = split_counts.get(s, 0) + 1
    ax.bar(split_counts.keys(), split_counts.values(), color=["blue", "purple", "brown"])
    ax.set_title("Split Distribution")
    ax.set_ylabel("Count")
    
    # Noise distribution
    ax = axes[0, 2]
    ax.hist(noise_ref, bins=30, alpha=0.5, label="Reference", color="blue")
    ax.hist(noise_search, bins=30, alpha=0.5, label="Search", color="red")
    ax.set_title("Noise Distribution (σ)")
    ax.set_xlabel("Gaussian σ")
    ax.legend()
    
    # Blur distribution
    ax = axes[1, 0]
    ax.hist(blur, bins=30, color="green", alpha=0.7)
    ax.set_title("Blur Distribution")
    ax.set_xlabel("Gaussian σ (pixels)")
    
    # Rotation distribution
    ax = axes[1, 1]
    ax.hist(rotation, bins=30, color="orange", alpha=0.7)
    ax.set_title("Rotation Distribution")
    ax.set_xlabel("|Angle| (degrees)")
    
    # Scale distribution
    ax = axes[1, 2]
    ax.hist(scale, bins=30, color="purple", alpha=0.7)
    ax.set_title("Scale Distribution")
    ax.set_xlabel("Scale Factor")
    
    # Contrast distribution
    ax = axes[2, 0]
    ax.hist(contrast, bins=30, color="cyan", alpha=0.7)
    ax.set_title("Contrast Distribution")
    ax.set_xlabel("Contrast Multiplier")
    
    # Thermal drift distribution
    ax = axes[2, 1]
    ax.hist(thermal, bins=30, color="brown", alpha=0.7)
    ax.set_title("Thermal Drift Distribution")
    ax.set_xlabel("Drift (pixels)")
    
    # Acoustic jitter distribution
    ax = axes[2, 2]
    ax.hist(acoustic, bins=30, color="pink", alpha=0.7)
    ax.set_title("Acoustic Jitter Distribution")
    ax.set_xlabel("Jitter Amplitude (pixels)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    logger.info(f"Statistics plots saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--output", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=25, help="Number of samples for contact sheet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--detail-sample", type=str, default=None, help="Create detail sheet for specific sample ID")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    annotations = load_annotations(dataset_dir)
    if not annotations:
        logger.error("Failed to load annotations")
        return
    
    logger.info(f"Loaded {len(annotations)} annotations")
    
    # Create contact sheet
    contact_path = output_dir / "contact_sheet.png"
    create_contact_sheet(dataset_dir, annotations, args.num_samples, contact_path, args.seed)
    
    # Create statistics plots
    stats_path = output_dir / "dataset_statistics.png"
    create_difficulty_distribution_plot(annotations, stats_path)
    
    # Create detail sheets for random samples
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(len(annotations), min(5, len(annotations)), replace=False)
    
    for idx in sample_indices:
        detail_path = output_dir / f"detail_{annotations[idx]['sample_id']}.png"
        create_sample_detail_sheet(dataset_dir, annotations[idx], detail_path)
    
    # Create detail for specific sample if requested
    if args.detail_sample:
        matching = [ann for ann in annotations if ann["sample_id"] == args.detail_sample]
        if matching:
            detail_path = output_dir / f"detail_{args.detail_sample}.png"
            create_sample_detail_sheet(dataset_dir, matching[0], detail_path)
        else:
            logger.error(f"Sample {args.detail_sample} not found")
    
    logger.info(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

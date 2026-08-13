# Drift-Sense Synthetic Dataset Generator

A comprehensive Python-based synthetic dataset generator for the Drift-Sense semiconductor image matching hackathon. Generates 1,000 high-quality DRAM-pattern SEM image pairs with realistic physical effects and complete ground-truth annotations.

## Quick Start

### Generate 30 test samples (fast):
```bash
python generate_synthetic_dataset.py --num-samples 30 --output test_dataset --seed 42
```

### Generate full 1,000-sample dataset:
```bash
python generate_synthetic_dataset.py --num-samples 1000 --output dataset --seed 42
```

### Visualize dataset:
```bash
python visualize_synthetic_dataset.py --dataset dataset --output visualizations --num-samples 25
```

---

## Dataset Overview

### Composition
- **Total**: 1,000 image pairs
- **Training**: 800 pairs (80%)
- **Validation**: 100 pairs (10%)
- **Test**: 100 pairs (10%)

### Image Format
- **Format**: 8-bit grayscale PNG
- **Search image size**: exactly 1000 × 1000 pixels
- **Reference image size**: 100–256 × 100–256 pixels (randomized)

### Ground Truth
Each sample includes precise annotations:
```json
{
  "sample_id": "000001",
  "split": "train",
  "reference_filename": "000001_ref.png",
  "search_filename": "000001_search.png",
  "reference_width": 150,
  "reference_height": 145,
  "bbox": {
    "x1": 320,
    "y1": 410,
    "x2": 470,
    "y2": 555
  },
  "center": {
    "x": 395.0,
    "y": 482.5
  },
  ...
}
```

---

## Architecture: DRAM Pattern

### Procedural Generation
Each reference and search image contains procedurally generated DRAM-style semiconductor patterns featuring:

- **Horizontal word-lines**: periodic features with controlled pitch
- **Vertical bit-lines**: periodic features perpendicular to word-lines
- **Contact/via dots**: small circular features at line intersections
- **Realistic imperfections**: slight offsets, missing contacts, line-width variations

### Key Parameters (per sample)
- **Pitch**: 8–16 pixels (line-to-line spacing)
- **Line width**: 1–3 pixels
- **Contact diameter**: 2–5 pixels
- **Number of lines**: 8–20 horizontal, 8–20 vertical
- **Spacing variation**: 0–15% irregularity

### No Template Reuse
Every reference and search pattern is independently generated with unique random parameters. Patterns are **not copies** but genuinely different architectures, preventing dataset trivial- ization.

---

## Degradation & Physical Effects

### Level 1: Architecture Randomization
Each sample uses different DRAM geometry (pitch, line width, contact size, etc.).

### Level 2: Geometric Imperfections
- Small random line offsets (±1–2 px)
- Occasional missing contacts (0–5%)
- Broken lines (0–3%)
- Line-width variations (±0–1 px)

### Level 3: Optical Effects (Reference Processing)
- **Rotation**: ±5° (varies per sample)
- **Scaling**: 0.85–1.15× (varies per sample)
- **Gaussian blur**: 0–2.5 pixel σ
- **Contrast**: 0.8–1.3× multiplier
- **Brightness**: ±15 intensity units

### Level 4: SEM Imaging Effects
- **Edge brightening**: SEM-style edge enhancement (0–1.0 strength)
- **Spatial intensity variation**: low-frequency brightness gradients (0–5%)
- **Gaussian noise**: 
  - Reference: σ = 2–15
  - Search: σ = 5–18 (higher due to lower magnification)
- **Poisson noise**: intensity-dependent shot noise (0–0.3× mean)

### Level 5: Physical Sensor/Stage Errors (NEW)

#### **Thermal Drift**
- Warping magnitude: 0–3 pixels
- Intensity variation: 0–8% brightness gradient
- Effect: Slow thermal expansion/contraction during SEM acquisition

#### **Acoustic Vibration**
- Jitter amplitude: 0–2 pixels
- Jitter frequency: 10–100 Hz
- Blur contribution: 0–1.5 pixel σ
- Effect: High-frequency mechanical stage vibrations

#### **Vibrational Creep**
- Drift vector: ±2–5 pixels over acquisition
- Creep rate: 0.1–1.0 pixels per time unit
- Non-rigid deformation applied
- Effect: Slow stage drift during image acquisition

---

## Difficulty Distribution

Dataset includes samples across difficulty levels to encourage robust model training:

### Easy (~25% of samples)
- Low noise (σ = 2–5)
- Low blur (σ = 0–0.8)
- Small rotation (±1–2°)
- High contrast (1.1–1.3×)
- Minimal thermal/acoustic effects
- Few defects

### Medium (~50% of samples)
- Moderate noise (σ = 5–10)
- Moderate blur (σ = 0.8–1.5)
- Moderate rotation (±2–4°)
- Moderate contrast (0.9–1.2×)
- Typical thermal/acoustic effects
- 1–3 defects per sample

### Hard (~25% of samples)
- High noise (σ = 10–15)
- Strong blur (σ = 1.5–2.5)
- Large rotation (±3–5°)
- Low contrast (0.8–1.0×)
- Significant thermal/acoustic/vibrational drift
- More defects and imperfections

---

## Directory Structure

```
dataset/
├── annotations.json              # All sample metadata + ground truth
├── generation_config.json        # Generation parameters & configuration
├── references/                   # High-magnification reference images
│   ├── 000000_ref.png
│   ├── 000001_ref.png
│   └── ...
└── searches/                     # Low-magnification search images (1000×1000)
    ├── 000000_search.png
    ├── 000001_search.png
    └── ...
```

---

## Metadata Format

### annotations.json Structure
Each entry contains:

```json
{
  "sample_id": "000001",
  "split": "train",                    # "train", "val", or "test"
  "reference_filename": "000001_ref.png",
  "search_filename": "000001_search.png",
  "reference_width": 150,
  "reference_height": 145,
  "search_width": 1000,
  "search_height": 1000,
  
  // Ground truth
  "bbox_x1": 320,
  "bbox_y1": 410,
  "bbox_x2": 470,
  "bbox_y2": 555,
  "center_x": 395.0,
  "center_y": 482.5,
  
  // DRAM architecture
  "pitch_px": 11.3,
  "line_width_px": 1.8,
  "contact_diameter_px": 3.2,
  "num_h_lines": 14,
  "num_v_lines": 15,
  
  // Transformations applied
  "rotation_deg": 2.5,
  "scale_factor": 0.98,
  "blur_sigma": 1.2,
  "contrast": 1.15,
  "brightness": 5,
  
  // Noise parameters
  "gauss_sigma_ref": 4.5,
  "gauss_sigma_search": 8.3,
  "poisson_intensity": 0.15,
  "spatial_variation_pct": 2.8,
  
  // Physical effects
  "thermal_drift_px": 1.2,
  "thermal_intensity_variation_pct": 3.5,
  "acoustic_jitter_amplitude_px": 0.8,
  "acoustic_jitter_frequency_hz": 45.0,
  "acoustic_blur_sigma": 0.5,
  "vibrational_drift_dx_px": -0.3,
  "vibrational_drift_dy_px": 1.1,
  "vibrational_creep_rate": 1.8,
  
  // Other
  "edge_brightening_strength": 0.6,
  "defect_count": 1,
  "defect_types": [],
  "difficulty_level": "medium",
  "random_seed": 10001
}
```

### generation_config.json
```json
{
  "num_samples": 1000,
  "num_train": 800,
  "num_val": 100,
  "num_test": 100,
  "seed": 42,
  "reference_size_range_px": [100, 256],
  "search_size_px": 1000,
  "architecture": "DRAM"
}
```

---

## Reproducibility

The dataset generation is **fully reproducible**. Same seed produces identical dataset:

```bash
# Same seed = identical dataset
python generate_synthetic_dataset.py --num-samples 1000 --seed 42

# Different seed = different samples
python generate_synthetic_dataset.py --num-samples 1000 --seed 123
```

### Seed Strategy
- **Master seed**: user-provided (default 42)
- **Split seeds**: derived from master seed
  - Training: `seed + 0`
  - Validation: `seed + 10000`
  - Test: `seed + 20000`
- **Sample seeds**: unique per sample within each split

This ensures:
- Complete reproducibility (same master seed)
- Different geometry per split (prevents information leakage)
- Each sample is independently randomized

---

## Quality Control

The generator automatically verifies:

1. ✓ Exactly N samples generated
2. ✓ Every reference image exists and is valid
3. ✓ Every search image exists and is valid
4. ✓ Every search image is exactly 1000 × 1000 pixels
5. ✓ Every annotation has valid bounding box
6. ✓ Every bounding box lies completely inside search image (with margin)
7. ✓ No duplicate filenames
8. ✓ Reference and search noise independently generated (never reused)
9. ✓ Ground truth matches actual transformed reference location
10. ✓ Train/validation/test use different seeds
11. ✓ No blank/corrupted images (mean pixel intensity > 10)

All checks pass before completion is reported.

---

## Visualization

### Contact Sheet
Create a 5×5 grid of 25 random samples with ground-truth bounding boxes:

```bash
python visualize_synthetic_dataset.py --dataset dataset --output viz --num-samples 25
```

Output: `visualizations/contact_sheet.png`

### Dataset Statistics
Generates plots showing:
- Difficulty distribution
- Split distribution
- Noise distribution (reference vs. search)
- Blur distribution
- Rotation distribution
- Scale distribution
- Contrast distribution
- Thermal drift
- Acoustic jitter

### Detail Sheets
Create detailed visualizations for individual samples showing:
- Full search image with ground-truth bounding box and center
- Zoomed ground-truth region
- Reference image
- Complete metadata table

```bash
python visualize_synthetic_dataset.py --dataset dataset --output viz --detail-sample 000001
```

---

## Command-Line Usage

### generate_synthetic_dataset.py

```bash
python generate_synthetic_dataset.py \
  --num-samples 1000 \
  --output dataset \
  --seed 42
```

**Arguments:**
- `--num-samples`: Number of samples to generate (default: 1000)
- `--output`: Output directory (default: "dataset")
- `--seed`: Random seed for reproducibility (default: 42)

### visualize_synthetic_dataset.py

```bash
python visualize_synthetic_dataset.py \
  --dataset dataset \
  --output visualizations \
  --num-samples 25 \
  --seed 42 \
  --detail-sample 000001
```

**Arguments:**
- `--dataset`: Dataset directory (default: "dataset")
- `--output`: Output directory for visualizations (default: "visualizations")
- `--num-samples`: Number of samples for contact sheet (default: 25)
- `--seed`: Random seed (default: 42)
- `--detail-sample`: Create detail sheet for specific sample ID (optional)

---

## Performance & Requirements

### System Requirements
- Python 3.8+
- 4–8 GB RAM (for 1,000-sample generation)
- ~5–10 minutes to generate 1,000 samples (depending on CPU)
- ~300 MB disk space for complete 1,000-sample dataset

### Dependencies
```
numpy>=1.21.0
opencv-python-headless>=4.5.0
pillow>=8.0.0
matplotlib>=3.3.0
scipy>=1.7.0
tqdm>=4.60.0
pandas>=1.3.0
```

Install via:
```bash
pip install -r requirements.txt
```

---

## Implementation Details

### DRAM Pattern Generation
Patterns are generated procedurally using:
- Line drawing with randomized positions
- Circular contact rendering at intersections
- Random defects (missing contacts, broken lines)
- Spacing variation to simulate realistic manufacturing variation

### Search-to-Reference Scale Relationship
- Reference: **1 nm/px** (1 µm field-of-view)
- Search: **10 nm/px** (10 µm field-of-view)
- Magnification ratio: **10:1**
- Reference appears ~100–150 px in final 1000 px search image

### SEM Edge Brightening
Not a generic sharpening filter. Instead:
1. Detect feature edges using Canny edge detector
2. Dilate edge mask slightly (1–3 px)
3. Add controlled brightness boost at edges (0–50 intensity units)
4. Maintains physical realism while enhancing contrast

### Independent Noise Generation
Absolutely critical—prevents trivial solutions:
- Reference noise generated once, independently
- Search noise generated separately, independently
- **Never reused** between reference and search
- Different magnitude (search typically higher due to lower mag)
- Different statistical properties possible

### Physical Error Modeling
- **Thermal drift**: modeled as spatial translation + low-freq intensity gradient
- **Acoustic jitter**: modeled as random pixel-level displacements + Poisson blur
- **Vibrational creep**: modeled as non-rigid elastic deformation field

---

## Output Example

```
INFO:generate_synthetic_dataset:Generating 1000 synthetic samples with seed 42...
INFO:generate_synthetic_dataset:Split: 800 train, 100 val, 100 test
Generating samples: 100%|██████████| 1000/1000 [08:23<00:00,  1.98it/s]
INFO:generate_synthetic_dataset:Annotations saved to dataset/annotations.json
INFO:generate_synthetic_dataset:Config saved to dataset/generation_config.json
INFO:generate_synthetic_dataset:Running quality control checks...
INFO:generate_synthetic_dataset:All quality checks PASSED ✓

============================================================
DATASET SUMMARY
============================================================
Total samples: 1000
Training: 800
Validation: 100
Test: 100
Image size: 1000 × 1000
Architecture: DRAM
Average noise (ref): 7.50
Average blur: 1.23
Average scale: 0.99
Average rotation: 2.65°
============================================================
```

---

## Known Limitations & Future Work

### Current Version
- Single architecture (DRAM) per generation run
- Reference patterns not rotated/scaled relative to search (only relative transformation)
- Defects tracked in metadata but not detailed

### Future Enhancements
- Multi-architecture support (FinFET, 3D NAND, etc.)
- More realistic SEM physics (secondary electron yield, charging effects)
- Automated defect classification
- Interactive visualization dashboard
- Streaming generation for very large datasets

---

## License & Citation

This dataset generator is part of the Drift-Sense semiconductor image matching hackathon.

---

## Questions?

For issues, feature requests, or questions:
1. Check existing samples via visualization scripts
2. Examine `annotations.json` for per-sample metadata
3. Review parameter distributions using statistics plots

---

Generated with ❤️ for the Drift-Sense hackathon.

#!/usr/bin/env python3
"""
Comprehensive synthetic dataset generator for Drift-Sense hackathon.

Generates 1000 DRAM-pattern SEM image pairs with:
- 800 training samples
- 100 validation samples  
- 100 test samples
- Full metadata and ground-truth annotations
- Physical SEM effects (thermal, acoustic, vibrational)
- Quality-control checks

Usage:
    python generate_synthetic_dataset.py --num-samples 1000 --output dataset --seed 42
    python generate_synthetic_dataset.py --num-samples 30 --output debug --seed 42
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# DRAM Architecture Generation
# ============================================================================

@dataclass
class DRAMParams:
    """DRAM architecture parameters."""
    pitch_px: float  # word-line/bit-line spacing
    line_width_px: float  # horizontal/vertical line width
    contact_diameter_px: float  # intersection dot diameter
    num_h_lines: int  # number of horizontal word-lines
    num_v_lines: int  # number of vertical bit-lines
    spacing_variation_pct: float  # line spacing irregularity
    
    def to_dict(self) -> dict:
        return asdict(self)


def generate_dram_reference(params: DRAMParams, ref_size_px: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a clean DRAM reference pattern.
    
    Args:
        params: DRAM architecture parameters
        ref_size_px: target reference image size (e.g., 100-256)
        rng: numpy random generator
        
    Returns:
        Clean DRAM reference image (ref_size_px x ref_size_px), uint8 grayscale
    """
    img = np.zeros((ref_size_px, ref_size_px), dtype=np.float32)
    
    # Calculate positions
    start_offset = (ref_size_px - params.num_h_lines * params.pitch_px) / 2.0
    
    # Draw horizontal word-lines
    for i in range(params.num_h_lines):
        y = int(start_offset + i * params.pitch_px)
        # Add small random variation
        y_offset = int(rng.normal(0, params.spacing_variation_pct * params.pitch_px / 100.0))
        y = np.clip(y + y_offset, 1, ref_size_px - 2)
        
        h = int(max(params.line_width_px, 1))
        img[max(0, y - h//2):min(ref_size_px, y + h//2 + 1), :] = 255
    
    # Draw vertical bit-lines
    start_offset_v = (ref_size_px - params.num_v_lines * params.pitch_px) / 2.0
    for j in range(params.num_v_lines):
        x = int(start_offset_v + j * params.pitch_px)
        # Add small random variation
        x_offset = int(rng.normal(0, params.spacing_variation_pct * params.pitch_px / 100.0))
        x = np.clip(x + x_offset, 1, ref_size_px - 2)
        
        w = int(max(params.line_width_px, 1))
        img[:, max(0, x - w//2):min(ref_size_px, x + w//2 + 1)] = 255
    
    # Draw contacts at intersections
    contact_rad = int(max(params.contact_diameter_px / 2.0, 1))
    for i in range(params.num_h_lines):
        for j in range(params.num_v_lines):
            y_base = int(start_offset + i * params.pitch_px)
            y_offset = int(rng.normal(0, params.spacing_variation_pct * params.pitch_px / 100.0))
            y = np.clip(y_base + y_offset, 1, ref_size_px - 2)
            
            x_base = int(start_offset_v + j * params.pitch_px)
            x_offset = int(rng.normal(0, params.spacing_variation_pct * params.pitch_px / 100.0))
            x = np.clip(x_base + x_offset, 1, ref_size_px - 2)
            
            # Skip some contacts (defects)
            if rng.random() > 0.03:  # 3% missing contacts
                cv2.circle(img, (x, y), contact_rad, 255, -1)
    
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_dram_search_base(
    params: DRAMParams,
    search_size_px: int,
    scale_factor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate the base DRAM pattern for search image (larger scale).
    
    The reference should appear ~100px in the final 1000px image,
    so we generate at ~10x lower pitch/magnification.
    
    Args:
        params: DRAM architecture parameters
        search_size_px: final search image size (1000)
        scale_factor: scale relative to reference (~10)
        rng: random generator
        
    Returns:
        Base DRAM pattern for search image
    """
    # Use scaled parameters for search
    search_params = DRAMParams(
        pitch_px=params.pitch_px * scale_factor,
        line_width_px=params.line_width_px * scale_factor,
        contact_diameter_px=params.contact_diameter_px * scale_factor,
        num_h_lines=max(3, int(params.num_h_lines / 2)),  # Fewer lines at lower mag
        num_v_lines=max(3, int(params.num_v_lines / 2)),
        spacing_variation_pct=params.spacing_variation_pct,
    )
    
    # Generate full search canvas
    img = np.zeros((search_size_px, search_size_px), dtype=np.float32)
    
    start_offset = (search_size_px - search_params.num_h_lines * search_params.pitch_px) / 2.0
    
    # Draw horizontal lines
    for i in range(search_params.num_h_lines):
        y = int(start_offset + i * search_params.pitch_px)
        y_offset = int(rng.normal(0, search_params.spacing_variation_pct * search_params.pitch_px / 100.0))
        y = np.clip(y + y_offset, 1, search_size_px - 2)
        
        h = int(max(search_params.line_width_px, 1))
        img[max(0, y - h//2):min(search_size_px, y + h//2 + 1), :] = 255
    
    # Draw vertical lines
    start_offset_v = (search_size_px - search_params.num_v_lines * search_params.pitch_px) / 2.0
    for j in range(search_params.num_v_lines):
        x = int(start_offset_v + j * search_params.pitch_px)
        x_offset = int(rng.normal(0, search_params.spacing_variation_pct * search_params.pitch_px / 100.0))
        x = np.clip(x + x_offset, 1, search_size_px - 2)
        
        w = int(max(search_params.line_width_px, 1))
        img[:, max(0, x - w//2):min(search_size_px, x + w//2 + 1)] = 255
    
    # Draw contacts
    contact_rad = int(max(search_params.contact_diameter_px / 2.0, 1))
    for i in range(search_params.num_h_lines):
        for j in range(search_params.num_v_lines):
            y_base = int(start_offset + i * search_params.pitch_px)
            y_offset = int(rng.normal(0, search_params.spacing_variation_pct * search_params.pitch_px / 100.0))
            y = np.clip(y_base + y_offset, 1, search_size_px - 2)
            
            x_base = int(start_offset_v + j * search_params.pitch_px)
            x_offset = int(rng.normal(0, search_params.spacing_variation_pct * search_params.pitch_px / 100.0))
            x = np.clip(x_base + x_offset, 1, search_size_px - 2)
            
            if rng.random() > 0.03:
                cv2.circle(img, (x, y), contact_rad, 255, -1)
    
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================================
# SEM Effects and Noise
# ============================================================================

def apply_edge_brightening(img: np.ndarray, strength: float) -> np.ndarray:
    """Apply SEM-style edge brightening."""
    if strength <= 0:
        return img
    
    # Compute edges
    edges = cv2.Canny(img, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    # Brighten edges
    result = img.astype(np.float32)
    result[edges > 0] = np.minimum(255, result[edges > 0] + strength * 50)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Apply Gaussian noise."""
    if sigma <= 0:
        return img
    noise = rng.normal(0, sigma, img.shape)
    result = img.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_poisson_noise(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Apply Poisson (shot) noise."""
    if intensity <= 0:
        return img
    
    # Scale intensity
    noise = rng.poisson(intensity * img.astype(np.float32) / 255.0)
    result = img.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_thermal_drift(
    img: np.ndarray,
    drift_magnitude_px: float,
    intensity_variation_pct: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply thermal drift effects: warping and intensity gradient."""
    h, w = img.shape
    result = img.astype(np.float32)
    
    if drift_magnitude_px > 0:
        # Create warping field (low-frequency)
        shift_x = int(rng.normal(0, drift_magnitude_px))
        shift_y = int(rng.normal(0, drift_magnitude_px))
        
        # Apply shift with clipping
        if shift_x != 0 or shift_y != 0:
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    if intensity_variation_pct > 0:
        # Add low-frequency intensity gradient
        y_coords = np.arange(h, dtype=np.float32) / h
        gradient = 1.0 + intensity_variation_pct * (y_coords.reshape(-1, 1) - 0.5) / 100.0
        result = result * gradient
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_acoustic_jitter(
    img: np.ndarray,
    jitter_amplitude_px: float,
    jitter_frequency_hz: float,
    blur_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply acoustic vibration effects: high-frequency jitter and blur."""
    if jitter_amplitude_px <= 0:
        return img
    
    h, w = img.shape
    result = img.astype(np.float32)
    
    # Add small random displacements
    displacement_x = int(rng.normal(0, jitter_amplitude_px))
    displacement_y = int(rng.normal(0, jitter_amplitude_px))
    
    if displacement_x != 0 or displacement_y != 0:
        M = np.float32([[1, 0, displacement_x], [0, 1, displacement_y]])
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Apply high-frequency blur
    if blur_sigma > 0.1:
        result = cv2.GaussianBlur(result, (3, 3), blur_sigma)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_vibrational_drift(
    img: np.ndarray,
    drift_vector_px: Tuple[float, float],
    creep_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply vibrational drift effects: non-rigid deformation."""
    h, w = img.shape
    dx, dy = drift_vector_px
    
    if abs(dx) < 0.1 and abs(dy) < 0.1:
        return img
    
    # Create elastic deformation field
    map_x = np.arange(w, dtype=np.float32)
    map_y = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(map_x, map_y)
    
    # Add creep-like distortion
    amplitude = creep_rate * 0.5
    map_x_new = map_x + amplitude * np.sin(2 * np.pi * map_y / h) * dx
    map_y_new = map_y + amplitude * np.sin(2 * np.pi * map_x / w) * dy
    
    result = cv2.remap(
        img.astype(np.float32),
        map_x_new,
        map_y_new,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================================
# Image Processing & Transformations
# ============================================================================

def apply_contrast_brightness(
    img: np.ndarray,
    contrast: float,
    brightness: int,
) -> np.ndarray:
    """Apply contrast and brightness adjustment."""
    result = img.astype(np.float32) * contrast + brightness
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_rotation_and_scale(
    img: np.ndarray,
    angle_deg: float,
    scale_factor: float,
) -> np.ndarray:
    """Apply rotation and scaling (keeping image size constant)."""
    h, w = img.shape
    center = (w / 2, h / 2)
    
    M = cv2.getRotationMatrix2D(center, angle_deg, scale_factor)
    result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return result


def apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur."""
    if sigma < 0.1:
        return img
    
    # Compute kernel size
    k = int(2 * round(3 * sigma) + 1)
    k = max(k, 3)
    if k % 2 == 0:
        k += 1
    
    return cv2.GaussianBlur(img, (k, k), sigma)


# ============================================================================
# Dataset Sample Generation
# ============================================================================

@dataclass
class SampleMetadata:
    """Metadata for a single sample."""
    sample_id: str
    split: str  # 'train', 'val', or 'test'
    reference_filename: str
    search_filename: str
    reference_width: int
    reference_height: int
    search_width: int = 1000
    search_height: int = 1000
    
    # Ground truth
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    center_x: float
    center_y: float
    
    # Architecture
    pitch_px: float
    line_width_px: float
    contact_diameter_px: float
    num_h_lines: int
    num_v_lines: int
    
    # Transformations
    rotation_deg: float
    scale_factor: float
    blur_sigma: float
    contrast: float
    brightness: int
    
    # Noise
    gauss_sigma_ref: float
    gauss_sigma_search: float
    poisson_intensity: float
    spatial_variation_pct: float
    
    # Physical effects
    thermal_drift_px: float
    thermal_intensity_variation_pct: float
    acoustic_jitter_amplitude_px: float
    acoustic_jitter_frequency_hz: float
    acoustic_blur_sigma: float
    vibrational_drift_dx_px: float
    vibrational_drift_dy_px: float
    vibrational_creep_rate: float
    
    # Other
    edge_brightening_strength: float
    defect_count: int
    defect_types: List[str]
    difficulty_level: str
    random_seed: int
    
    def to_dict(self) -> dict:
        return asdict(self)


def sample_difficulty_parameters(difficulty: str, rng: np.random.Generator) -> Dict:
    """Sample parameters based on difficulty level."""
    params = {}
    
    if difficulty == "easy":
        params["gauss_sigma_ref"] = rng.uniform(2, 5)
        params["gauss_sigma_search"] = rng.uniform(5, 10)
        params["blur_sigma"] = rng.uniform(0, 0.8)
        params["rotation_deg"] = rng.uniform(-1.5, 1.5)
        params["contrast"] = rng.uniform(1.1, 1.3)
        params["thermal_drift_px"] = rng.uniform(0, 0.5)
        params["acoustic_jitter_amplitude_px"] = rng.uniform(0, 0.5)
        params["vibrational_creep_rate"] = rng.uniform(0, 1)
        params["defect_count"] = rng.integers(0, 2)
        
    elif difficulty == "medium":
        params["gauss_sigma_ref"] = rng.uniform(5, 10)
        params["gauss_sigma_search"] = rng.uniform(8, 12)
        params["blur_sigma"] = rng.uniform(0.8, 1.5)
        params["rotation_deg"] = rng.uniform(-3, 3)
        params["contrast"] = rng.uniform(0.9, 1.2)
        params["thermal_drift_px"] = rng.uniform(0.5, 1.5)
        params["acoustic_jitter_amplitude_px"] = rng.uniform(0.5, 1.2)
        params["vibrational_creep_rate"] = rng.uniform(1, 3)
        params["defect_count"] = rng.integers(1, 3)
        
    else:  # hard
        params["gauss_sigma_ref"] = rng.uniform(10, 15)
        params["gauss_sigma_search"] = rng.uniform(12, 18)
        params["blur_sigma"] = rng.uniform(1.5, 2.5)
        params["rotation_deg"] = rng.uniform(-5, 5)
        params["contrast"] = rng.uniform(0.75, 1.0)
        params["thermal_drift_px"] = rng.uniform(1.5, 3)
        params["acoustic_jitter_amplitude_px"] = rng.uniform(1.2, 2)
        params["vibrational_creep_rate"] = rng.uniform(3, 5)
        params["defect_count"] = rng.integers(2, 5)
    
    # Common to all
    params["scale_factor"] = rng.uniform(0.85, 1.15)
    params["brightness"] = int(rng.uniform(-10, 10))
    params["thermal_intensity_variation_pct"] = rng.uniform(0, 5)
    params["acoustic_jitter_frequency_hz"] = rng.uniform(10, 100)
    params["acoustic_blur_sigma"] = rng.uniform(0, 1)
    params["edge_brightening_strength"] = rng.uniform(0, 0.8)
    params["poisson_intensity"] = rng.uniform(0, 0.3)
    params["spatial_variation_pct"] = rng.uniform(0, 5)
    
    return params


def generate_sample(
    sample_id: str,
    split: str,
    output_dir: Path,
    seed: int,
    difficulty: str = None,
) -> Tuple[SampleMetadata, bool]:
    """
    Generate a single sample.
    
    Returns:
        (metadata, success)
    """
    rng = np.random.default_rng(seed)
    
    # Sample difficulty if not specified
    if difficulty is None:
        rand = rng.random()
        if rand < 0.25:
            difficulty = "easy"
        elif rand < 0.75:
            difficulty = "medium"
        else:
            difficulty = "hard"
    
    try:
        # Sample architecture parameters
        dram_params = DRAMParams(
            pitch_px=rng.uniform(8, 16),
            line_width_px=rng.uniform(1, 3),
            contact_diameter_px=rng.uniform(2, 5),
            num_h_lines=rng.integers(8, 21),
            num_v_lines=rng.integers(8, 21),
            spacing_variation_pct=rng.uniform(0, 15),
        )
        
        # Sample processing parameters
        proc_params = sample_difficulty_parameters(difficulty, rng)
        
        # Generate reference image
        ref_size_px = rng.integers(100, 257)
        reference = generate_dram_reference(dram_params, ref_size_px, rng)
        
        # Apply reference processing
        reference = apply_edge_brightening(reference, proc_params["edge_brightening_strength"])
        reference = apply_rotation_and_scale(
            reference,
            proc_params["rotation_deg"],
            proc_params["scale_factor"],
        )
        reference = apply_gaussian_blur(reference, proc_params["blur_sigma"])
        reference = apply_contrast_brightness(
            reference,
            proc_params["contrast"],
            proc_params["brightness"],
        )
        reference = apply_gaussian_noise(reference, proc_params["gauss_sigma_ref"], rng)
        reference = apply_poisson_noise(reference, proc_params["poisson_intensity"], rng)
        reference = apply_thermal_drift(
            reference,
            proc_params["thermal_drift_px"],
            proc_params["thermal_intensity_variation_pct"],
            rng,
        )
        reference = apply_acoustic_jitter(
            reference,
            proc_params["acoustic_jitter_amplitude_px"],
            proc_params["acoustic_jitter_frequency_hz"],
            proc_params["acoustic_blur_sigma"],
            rng,
        )
        
        ref_h, ref_w = reference.shape
        
        # Generate search image base
        search = generate_dram_search_base(dram_params, 1000, 10.0, rng)
        
        # Insert reference into search at random location
        # Calculate valid insertion bounds
        max_y = max(0, 1000 - ref_h)
        max_x = max(0, 1000 - ref_w)
        
        if max_y <= 0 or max_x <= 0:
            # Reference is too large or equal to search
            return None, False
        
        insert_y = rng.integers(0, max_y + 1)
        insert_x = rng.integers(0, max_x + 1)
        
        # Create combined search with reference inserted
        search_combined = search.astype(np.float32)
        search_combined[insert_y:insert_y + ref_h, insert_x:insert_x + ref_w] = reference.astype(np.float32) * 0.7 + search[insert_y:insert_y + ref_h, insert_x:insert_x + ref_w] * 0.3
        
        # Calculate ground truth before search processing
        bbox_y1 = insert_y
        bbox_x1 = insert_x
        bbox_y2 = min(insert_y + ref_h, 1000)
        bbox_x2 = min(insert_x + ref_w, 1000)
        center_x = (bbox_x1 + bbox_x2) / 2.0
        center_y = (bbox_y1 + bbox_y2) / 2.0
        
        # Apply search-specific effects (higher noise, more blur)
        search_combined = np.clip(search_combined, 0, 255).astype(np.uint8)
        search_combined = apply_edge_brightening(search_combined, proc_params["edge_brightening_strength"] * 0.5)
        search_combined = apply_gaussian_noise(search_combined, proc_params["gauss_sigma_search"], rng)
        search_combined = apply_poisson_noise(search_combined, proc_params["poisson_intensity"] * 1.5, rng)
        search_combined = apply_thermal_drift(
            search_combined,
            proc_params["thermal_drift_px"],
            proc_params["thermal_intensity_variation_pct"],
            rng,
        )
        search_combined = apply_acoustic_jitter(
            search_combined,
            proc_params["acoustic_jitter_amplitude_px"],
            proc_params["acoustic_jitter_frequency_hz"],
            proc_params["acoustic_blur_sigma"],
            rng,
        )
        vibrational_drift = (rng.uniform(-3, 3), rng.uniform(-3, 3))
        search_combined = apply_vibrational_drift(
            search_combined,
            vibrational_drift,
            proc_params["vibrational_creep_rate"],
            rng,
        )
        
        # Verify search dimensions
        assert search_combined.shape == (1000, 1000), f"Search shape {search_combined.shape} != (1000, 1000)"
        
        # Save images
        ref_path = output_dir / "references" / f"{sample_id}_ref.png"
        search_path = output_dir / "searches" / f"{sample_id}_search.png"
        
        cv2.imwrite(str(ref_path), reference)
        cv2.imwrite(str(search_path), search_combined)
        
        # Create metadata
        metadata = SampleMetadata(
            sample_id=sample_id,
            split=split,
            reference_filename=ref_path.name,
            search_filename=search_path.name,
            reference_width=ref_w,
            reference_height=ref_h,
            bbox_x1=bbox_x1,
            bbox_y1=bbox_y1,
            bbox_x2=bbox_x2,
            bbox_y2=bbox_y2,
            center_x=center_x,
            center_y=center_y,
            pitch_px=dram_params.pitch_px,
            line_width_px=dram_params.line_width_px,
            contact_diameter_px=dram_params.contact_diameter_px,
            num_h_lines=dram_params.num_h_lines,
            num_v_lines=dram_params.num_v_lines,
            rotation_deg=proc_params["rotation_deg"],
            scale_factor=proc_params["scale_factor"],
            blur_sigma=proc_params["blur_sigma"],
            contrast=proc_params["contrast"],
            brightness=proc_params["brightness"],
            gauss_sigma_ref=proc_params["gauss_sigma_ref"],
            gauss_sigma_search=proc_params["gauss_sigma_search"],
            poisson_intensity=proc_params["poisson_intensity"],
            spatial_variation_pct=proc_params["spatial_variation_pct"],
            thermal_drift_px=proc_params["thermal_drift_px"],
            thermal_intensity_variation_pct=proc_params["thermal_intensity_variation_pct"],
            acoustic_jitter_amplitude_px=proc_params["acoustic_jitter_amplitude_px"],
            acoustic_jitter_frequency_hz=proc_params["acoustic_jitter_frequency_hz"],
            acoustic_blur_sigma=proc_params["acoustic_blur_sigma"],
            vibrational_drift_dx_px=vibrational_drift[0],
            vibrational_drift_dy_px=vibrational_drift[1],
            vibrational_creep_rate=proc_params["vibrational_creep_rate"],
            edge_brightening_strength=proc_params["edge_brightening_strength"],
            defect_count=proc_params["defect_count"],
            defect_types=[],  # TODO: track actual defects
            difficulty_level=difficulty,
            random_seed=seed,
        )
        
        return metadata, True
        
    except Exception as e:
        logger.error(f"Failed to generate sample {sample_id}: {e}")
        return None, False


# ============================================================================
# Quality Control
# ============================================================================

def run_quality_checks(
    output_dir: Path,
    annotations: List[Dict],
    num_samples: int,
) -> bool:
    """Run comprehensive quality checks."""
    logger.info("Running quality control checks...")
    issues = []
    
    # Check 1: Total count
    if len(annotations) != num_samples:
        issues.append(f"Expected {num_samples} samples, found {len(annotations)}")
    
    # Check 2-10: Per-sample validation
    for i, ann in enumerate(annotations):
        sample_id = ann.get("sample_id", f"unknown_{i}")
        
        # Check reference exists
        ref_file = output_dir / "references" / ann["reference_filename"]
        if not ref_file.exists():
            issues.append(f"{sample_id}: reference file missing")
            continue
        
        # Check search exists
        search_file = output_dir / "searches" / ann["search_filename"]
        if not search_file.exists():
            issues.append(f"{sample_id}: search file missing")
            continue
        
        # Check search dimensions
        search_img = cv2.imread(str(search_file), cv2.IMREAD_GRAYSCALE)
        if search_img.shape != (1000, 1000):
            issues.append(f"{sample_id}: search shape {search_img.shape} != (1000, 1000)")
        
        # Check bounding box validity
        bbox = ann.get("bbox", {})
        x1, y1, x2, y2 = bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")
        
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            issues.append(f"{sample_id}: invalid bounding box ({x1}, {y1}, {x2}, {y2})")
        
        # Check center is within bbox
        center = ann.get("center", {})
        cx, cy = center.get("x"), center.get("y")
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            issues.append(f"{sample_id}: center ({cx}, {cy}) outside bbox")
        
        # Check image not blank
        ref_img = cv2.imread(str(ref_file), cv2.IMREAD_GRAYSCALE)
        if ref_img is None or np.mean(ref_img) < 10:
            issues.append(f"{sample_id}: reference appears blank")
        
        if search_img is None or np.mean(search_img) < 10:
            issues.append(f"{sample_id}: search appears blank")
    
    # Report
    if issues:
        logger.error(f"Quality checks FAILED with {len(issues)} issues:")
        for issue in issues[:20]:  # Print first 20
            logger.error(f"  - {issue}")
        if len(issues) > 20:
            logger.error(f"  ... and {len(issues) - 20} more")
        return False
    else:
        logger.info("All quality checks PASSED ✓")
        return True


# ============================================================================
# Main Dataset Generation
# ============================================================================

def generate_dataset(
    num_samples: int,
    output_dir: Path,
    seed: int,
) -> bool:
    """Generate complete dataset."""
    logger.info(f"Generating {num_samples} synthetic samples with seed {seed}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "references").mkdir(exist_ok=True)
    (output_dir / "searches").mkdir(exist_ok=True)
    
    # Split samples
    num_train = int(0.8 * num_samples)
    num_val = int(0.1 * num_samples)
    num_test = num_samples - num_train - num_val
    
    logger.info(f"Split: {num_train} train, {num_val} val, {num_test} test")
    
    all_metadata = []
    splits = {
        "train": (0, num_train, 0),
        "val": (num_train, num_train + num_val, 10000),
        "test": (num_train + num_val, num_samples, 20000),
    }
    
    total_generated = 0
    pbar = tqdm(total=num_samples, desc="Generating samples")
    
    for split_name, (start_idx, end_idx, seed_offset) in splits.items():
        for idx in range(start_idx, end_idx):
            sample_id = f"{idx:06d}"
            sample_seed = seed + seed_offset + idx
            
            metadata, success = generate_sample(
                sample_id,
                split_name,
                output_dir,
                sample_seed,
            )
            
            if success:
                all_metadata.append(metadata.to_dict())
                total_generated += 1
            
            pbar.update(1)
    
    pbar.close()
    
    # Save annotations
    annotations_file = output_dir / "annotations.json"
    with open(annotations_file, "w") as f:
        json.dump(all_metadata, f, indent=2)
    logger.info(f"Annotations saved to {annotations_file}")
    
    # Save config
    config = {
        "num_samples": num_samples,
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "seed": seed,
        "reference_size_range_px": [100, 256],
        "search_size_px": 1000,
        "architecture": "DRAM",
    }
    config_file = output_dir / "generation_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_file}")
    
    # Quality checks
    success = run_quality_checks(output_dir, all_metadata, total_generated)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {total_generated}")
    logger.info(f"Training: {len([m for m in all_metadata if m['split'] == 'train'])}")
    logger.info(f"Validation: {len([m for m in all_metadata if m['split'] == 'val'])}")
    logger.info(f"Test: {len([m for m in all_metadata if m['split'] == 'test'])}")
    logger.info(f"Image size: 1000 × 1000")
    logger.info(f"Architecture: DRAM")
    
    if all_metadata:
        noise_vals = [m["gauss_sigma_ref"] for m in all_metadata]
        blur_vals = [m["blur_sigma"] for m in all_metadata]
        scale_vals = [m["scale_factor"] for m in all_metadata]
        rot_vals = [abs(m["rotation_deg"]) for m in all_metadata]
        
        logger.info(f"Average noise (ref): {np.mean(noise_vals):.2f}")
        logger.info(f"Average blur: {np.mean(blur_vals):.2f}")
        logger.info(f"Average scale: {np.mean(scale_vals):.2f}")
        logger.info(f"Average rotation: {np.mean(rot_vals):.2f}°")
    
    logger.info("=" * 60)
    
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    success = generate_dataset(args.num_samples, output_dir, args.seed)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

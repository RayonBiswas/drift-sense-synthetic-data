"""
End-to-end generation of one sample, with exact ground truth.

Geometry of the problem
-----------------------
One world unit == one pixel of the final 1000 x 1000 search image.

  * The specimen is rasterized once, onto a fine canvas of 1000 * S pixels.
  * The **search image** is that whole canvas area-averaged down to 1000 x 1000
    (an S-fold reduction in magnification), then put through its own capture.
  * The **reference image** is a rotated square window of the *same* fine canvas,
    F = 100 * scale world units on a side, cut at full supersampled resolution
    and stored at its own randomized size of 100-256 px. It is therefore a
    genuinely higher-magnification view of the same device, not an upscaled crop
    of the search image, and it is put through a completely separate capture.

Reference sourcing: the reference is *cut from* the search's own layout rather
than pasted into it. Pasting a foreign patch leaves a seam and makes the
surrounding context inconsistent with the patch; cutting gives a seam-free image
and a ground truth that is exact by construction.

Ground truth
------------
Never measured from the output images. The four corners of the reference window
are known analytically in world coordinates, and are then pushed through the
*same* thermal-drift and vibration displacement functions that warp the search
image. The stored box is the axis-aligned bounding box of those four mapped
corners; the exact rotated quadrilateral is stored alongside it.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import sem
from . import random as artifacts
from .layout import render_fine_canvas
from .params import SEARCH_SIZE_PX, sample_parameters


def _reference_window(fine: np.ndarray, params: dict, s: int) -> tuple:
    """Cut the rotated reference window out of the fine canvas.

    Returns (window, corners_world). `corners_world` are the four corners of the
    window expressed in world/search coordinates, in the order
    (top-left, top-right, bottom-right, bottom-left) *of the reference image*.
    """
    footprint = params["footprint_px"]
    theta_deg = params["rotation_deg"]
    cx_f = params["position_x"] * s
    cy_f = params["position_y"] * s
    side_f = max(int(round(footprint * s)), 8)

    # dst = A * (src - c) + centre, with A the linear part of the OpenCV
    # rotation matrix. Inverting A maps reference-image offsets back to world
    # offsets, which is exactly what the ground-truth corners need.
    m = cv2.getRotationMatrix2D((float(cx_f), float(cy_f)), float(theta_deg), 1.0)
    m[0, 2] += side_f / 2.0 - cx_f
    m[1, 2] += side_f / 2.0 - cy_f
    window = cv2.warpAffine(fine, m, (side_f, side_f),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    th = np.deg2rad(theta_deg)
    a_inv = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]], dtype=np.float64)
    half = footprint / 2.0
    local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    corners_world = local @ a_inv.T + np.array([params["position_x"], params["position_y"]])
    return window, corners_world


def build_sample(sample_seed: int, difficulty: str, supersample: int = 10,
                 architecture: str = "dram") -> dict:
    """Generate one complete sample: both images, the ground truth and the
    full parameter record.

    `architecture` is "dram" or "finfet". It changes only what is rasterized
    onto the specimen canvas -- the imaging chain, the ground-truth algebra and
    the random draw sequence are identical for both.
    """
    params = sample_parameters(sample_seed, difficulty, architecture)
    seeds = params["seeds"]
    s = int(supersample)

    fine, defect_counts = render_fine_canvas(params, s, seeds["geometry_seed"])

    # Cut the reference window from the clean specimen before the search-side
    # capture touches anything.
    window, corners_world = _reference_window(fine, params, s)

    ref_img, _, _, ref_extra = sem.image_capture(
        window,
        out_size=params["reference_size_px"],
        imaging=params["reference_imaging"],
        noise_seed=seeds["reference_noise_seed"],
        scan_seed=seeds["reference_scan_seed"],
        local_blur=(params["defect_density"] > 0 and (sample_seed % 7 == 0)),
    )
    del window

    search_img, dx, dy, search_extra = sem.image_capture(
        fine,
        out_size=SEARCH_SIZE_PX,
        imaging=params["search_imaging"],
        noise_seed=seeds["search_noise_seed"],
        scan_seed=seeds["search_scan_seed"],
        local_blur=(params["defect_density"] > 0 and (sample_seed % 5 == 0)),
    )
    del fine

    # Push the ground truth through the same geometric distortions as the image,
    # in the same order: row displacement first, then radial barrel distortion.
    quad = sem.forward_map_points(corners_world, dx, dy)
    centre_pt = sem.forward_map_points(
        np.array([[params["position_x"], params["position_y"]]]), dx, dy)

    barrel_k = float(params["search_imaging"].get("barrel_k", 0.0))
    if barrel_k != 0.0:
        quad = artifacts.barrel_forward_map_points(
            quad, barrel_k, SEARCH_SIZE_PX, SEARCH_SIZE_PX)
        centre_pt = artifacts.barrel_forward_map_points(
            centre_pt, barrel_k, SEARCH_SIZE_PX, SEARCH_SIZE_PX)
    centre = centre_pt[0]

    x1, y1 = quad[:, 0].min(), quad[:, 1].min()
    x2, y2 = quad[:, 0].max(), quad[:, 1].max()

    return {
        "reference_img": ref_img,
        "search_img": search_img,
        "params": params,
        "defect_counts": defect_counts,
        "reference_extra": ref_extra,
        "search_extra": search_extra,
        "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
        "center": {"x": float(centre[0]), "y": float(centre[1])},
        "quad": [[float(a), float(b)] for a, b in quad],
    }


def annotate(sample: dict, image_id: str, split: str,
             reference_filename: str, search_filename: str) -> dict:
    """Flatten one sample into its annotations.json record."""
    p = sample["params"]
    ref_img = sample["reference_img"]
    search_img = sample["search_img"]

    defects = dict(sample["defect_counts"])
    defects["local_blur_reference"] = sample["reference_extra"].get("local_blur")
    defects["local_blur_search"] = sample["search_extra"].get("local_blur")

    return {
        "image_id": image_id,
        "split": split,
        # Which device family was rasterized: "dram" or "finfet". Named
        # `architecture_kind` because `architecture` below is already taken by
        # the P01-P08 geometry block, and datasets generated before FinFET
        # existed must stay readable.
        "architecture_kind": p.get("architecture", "dram"),
        "reference_filename": reference_filename,
        "search_filename": search_filename,

        "reference_width": int(ref_img.shape[1]),
        "reference_height": int(ref_img.shape[0]),
        "search_width": int(search_img.shape[1]),
        "search_height": int(search_img.shape[0]),

        "bbox": sample["bbox"],
        "center": sample["center"],
        "quad": sample["quad"],

        "difficulty": p["difficulty"],
        "random_seed": int(p["sample_seed"]),
        "seeds": p["seeds"],

        # P01-P08: what the device looks like
        "architecture": {
            "P01_word_line_pitch": round(p["word_line_pitch"], 5),
            "P02_bit_to_word_pitch_ratio": round(p["bit_to_word_pitch_ratio"], 5),
            "bit_line_pitch": round(p["bit_line_pitch"], 5),
            "P03_line_width_frac": round(p["line_width_frac"], 5),
            "P04_contact_diameter_frac": round(p["contact_diameter_frac"], 5),
            "P05_block_line_count": int(p["block_line_count"]),
            "block_size": round(p["block_size"], 4),
            "strip_width": round(p["strip_width"], 4),
            "P06_spacing_jitter": round(p["spacing_jitter"], 5),
            "P07_phase_offset": [round(v, 5) for v in p["phase_offset"]],
            "P08_defect_density": round(p["defect_density"], 5),
        },

        # P09/P10/P17/P18: how the reference relates to the search
        "P09_scale": round(p["scale"], 5),
        "P10_rotation_deg": round(p["rotation_deg"], 5),
        "footprint_px": round(p["footprint_px"], 4),
        "P17_position_x": round(p["position_x"], 4),
        "P18_position_y": round(p["position_y"], 4),
        # "drift"   the stage aimed at the site and landed slightly off, so the
        #           target is near the centre of the fresh scan
        # "uniform" the site was lost entirely and must be re-acquired from the
        #           whole frame
        "position_mode": p["position_mode"],
        "reference_to_search_px_ratio": round(
            p["reference_size_px"] / p["footprint_px"], 5),

        # P11-P16, P19, P20: drawn separately for the two captures
        "reference_imaging": {k: (round(v, 5) if isinstance(v, float) else v)
                              for k, v in p["reference_imaging"].items()},
        "search_imaging": {k: (round(v, 5) if isinstance(v, float) else v)
                           for k, v in p["search_imaging"].items()},

        "defects": defects,
    }


# --------------------------------------------------------------------------- #
# Worker entry point
# --------------------------------------------------------------------------- #

def render_one_sample(job: dict) -> dict:
    """Generate one sample and write both PNGs. Module-level and
    dict-argument so it is picklable for multiprocessing on Windows (spawn)."""
    import os

    sample = build_sample(job["sample_seed"], job["difficulty"], job["supersample"],
                          job.get("architecture", "dram"))

    ref_name = f"{job['image_id']}.png"
    search_name = f"{job['image_id']}.png"
    ref_path = os.path.join(job["reference_dir"], ref_name)
    search_path = os.path.join(job["search_dir"], search_name)

    # PNG level 3: the noisy 1000x1000 frames compress poorly, and pushing the
    # level higher costs far more time than it saves bytes.
    flags = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if not cv2.imwrite(ref_path, sample["reference_img"], flags):
        raise IOError(f"failed to write {ref_path}")
    if not cv2.imwrite(search_path, sample["search_img"], flags):
        raise IOError(f"failed to write {search_path}")

    record = annotate(sample, job["image_id"], job["split"],
                      f"references/{ref_name}", f"searches/{search_name}")
    record["_stats"] = {
        "reference_std": float(sample["reference_img"].std()),
        "search_std": float(sample["search_img"].std()),
        "reference_mean": float(sample["reference_img"].mean()),
        "search_mean": float(sample["search_img"].mean()),
    }
    return record
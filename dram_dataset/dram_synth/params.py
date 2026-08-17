"""
The randomized generation parameters.

Every sample is described by an explicit, enumerated parameter set. P01-P18 are
the parameters implied by the five-level randomization hierarchy of the spec;
P19 and P20 are the two additional SEM scan-error models (thermal drift and
vibration).

    Level 1 - Architecture
      P01 word_line_pitch          word-line pitch, in world units
      P02 bit_to_word_pitch_ratio  bit-line pitch as a multiple of P01 (~1.5 => 6F^2)
      P03 line_width_frac          line width as a fraction of its own pitch
      P04 contact_diameter_frac    contact diameter as a fraction of P01
      P05 block_line_count         word lines per array block ("mat")

    Level 2 - Geometry
      P06 spacing_jitter           per-line pitch perturbation, std in world units
      P07 phase_offset             global (x, y) lattice phase, fraction of pitch
      P08 defect_density           per-feature defect probability (often exactly 0)

    Level 3 - Imaging
      P09 scale                    reference footprint = 100 * scale world units
      P10 rotation_deg             reference rotation relative to the search image
      P11 blur_sigma               Gaussian optical blur sigma (per image)
      P12 brightness               additive brightness offset (per image)
      P13 contrast                 multiplicative contrast gain (per image)

    Level 4 - SEM effects
      P14 edge_gain                SEM edge-brightening strength (per image)
      P15 shading_amplitude        low-frequency spatial intensity variation (per image)
      P16 sensor_noise             Gaussian sigma + Poisson dose (per image)

    Level 5 - Placement
      P17 position_x               reference centre x, in world units
      P18 position_y               reference centre y, in world units

    Additional scan-error models
      P19 thermal_drift            total drift accumulated over one frame + direction
      P20 vibration                amplitude, frequency, phase and jitter of stage vibration

P11-P16, P19 and P20 are sampled *independently for the reference and the search
image*, because the two are separate physical captures. The search side is drawn
from consistently harsher ranges: it is the fast, wide-area scan.

One world unit == one pixel of the final 1000x1000 search image, so P01, P09,
P17 and P18 can be read directly as search-image pixels.
"""

from __future__ import annotations

import numpy as np

from .random import draw_profile

# --------------------------------------------------------------------------- #
# Fixed geometry of the problem
# --------------------------------------------------------------------------- #

SEARCH_SIZE_PX = 1000          # final search image is exactly 1000 x 1000
NOMINAL_FOOTPRINT_PX = 100.0   # reference occupies ~100 x 100 px of the search
REFERENCE_SIZE_RANGE = (100, 256)   # stored reference image side length, px

DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_MIX = {"easy": 0.25, "medium": 0.50, "hard": 0.25}

# --------------------------------------------------------------------------- #
# Placement: where the stored site lands in the fresh scan
# --------------------------------------------------------------------------- #
# This is navigation-error recovery, not a hunt for a patch dropped anywhere in
# the frame. The tool commands a move to a site it has visited before and the
# stage lands slightly off, so on a revisit the target sits *near the centre* of
# the freshly captured wide field. Each stage axis contributes an independent,
# roughly zero-mean positioning error, which makes the per-axis miss Gaussian
# and the radial miss Rayleigh-distributed.
#
# Placing the site uniformly across the frame -- which is what this generator
# did originally -- models a different and less physical task, and it makes the
# problem statement's "return the match closest to the centre of the Search
# Image" rule carry no information at all. A minority of samples keep uniform
# placement to cover the worst case, where the tool has lost the site entirely
# and has to re-acquire it from the whole frame.
DRIFT_FRACTION = 0.70          # share of samples that model a normal revisit
DRIFT_SIGMA_PX = 110.0         # per-axis stage positioning error, search px
DRIFT_TRUNCATE_SIGMA = 3.0     # past this the move is a gross fault, not drift

# --------------------------------------------------------------------------- #
# Level 1 / Level 2 ranges (architecture + geometry; difficulty-independent)
# --------------------------------------------------------------------------- #

ARCH_RANGES = {
    "word_line_pitch": (6.0, 14.0),          # P01, world units
    "bit_to_word_pitch_ratio": (1.30, 1.80),  # P02
    "line_width_frac": (0.34, 0.54),          # P03
    "contact_diameter_frac": (0.30, 0.62),    # P04
    "block_size": (90.0, 320.0),              # world units; P05 derives from this
    "strip_width": (6.0, 18.0),               # peripheral/routing strip width
}

GEOM_RANGES = {
    "spacing_jitter": (0.02, 0.35),           # P06, world units
}

# P08: probability a sample has *any* defects, and the density if it does.
DEFECT_RANGES = {
    "easy":   {"p_any": 0.20, "density": (0.004, 0.015)},
    "medium": {"p_any": 0.50, "density": (0.010, 0.035)},
    "hard":   {"p_any": 0.75, "density": (0.030, 0.080)},
}

# --------------------------------------------------------------------------- #
# Level 3 / Level 4 / P19 / P20 ranges, per difficulty tier
# `ref` and `search` are drawn independently from their own ranges.
# --------------------------------------------------------------------------- #

TIERS = {
    "easy": {
        "scale": (0.95, 1.05),                 # P09
        "rotation_deg": (-1.5, 1.5),           # P10
        "blur_sigma": {"ref": (0.35, 0.70), "search": (0.50, 0.95)},      # P11
        "brightness": {"ref": (-8, 8), "search": (-10, 10)},              # P12
        "contrast": {"ref": (0.95, 1.08), "search": (0.92, 1.10)},        # P13
        "edge_gain": {"ref": (0.20, 0.55), "search": (0.18, 0.50)},       # P14
        "shading_amplitude": {"ref": (0.02, 0.06), "search": (0.03, 0.08)},  # P15
        "noise_sigma": {"ref": (1.0, 3.0), "search": (2.0, 5.0)},         # P16
        "dose": {"ref": (1500, 4000), "search": (500, 1500)},             # P16
        "thermal_drift": {"ref": (0.00, 0.30), "search": (0.20, 1.00)},   # P19
        "vibration_amp": {"ref": (0.00, 0.15), "search": (0.05, 0.35)},   # P20
        "salt_pepper_prob": {"ref": (0.0, 0.0005), "search": (0.0, 0.001)},  # impulse
        "impulse_burst_prob": {"ref": (0.0, 0.0), "search": (0.0, 0.0)},
        "brownian_sigma": {"ref": (0.0, 0.0), "search": (0.0, 0.0)},
    },
    "medium": {
        "scale": (0.90, 1.10),
        "rotation_deg": (-3.0, 3.0),
        "blur_sigma": {"ref": (0.50, 0.95), "search": (0.80, 1.55)},
        "brightness": {"ref": (-18, 18), "search": (-22, 22)},
        "contrast": {"ref": (0.85, 1.18), "search": (0.80, 1.22)},
        "edge_gain": {"ref": (0.15, 0.60), "search": (0.15, 0.55)},
        "shading_amplitude": {"ref": (0.05, 0.13), "search": (0.06, 0.16)},
        "noise_sigma": {"ref": (2.0, 5.0), "search": (4.0, 9.0)},
        "dose": {"ref": (700, 2000), "search": (150, 600)},
        "thermal_drift": {"ref": (0.10, 0.60), "search": (0.80, 2.50)},
        "vibration_amp": {"ref": (0.05, 0.30), "search": (0.20, 0.80)},
        "salt_pepper_prob": {"ref": (0.0003, 0.001), "search": (0.001, 0.003)},  # impulse
        "impulse_burst_prob": {"ref": (0.0, 0.0001), "search": (0.0001, 0.0005)},
        "brownian_sigma": {"ref": (0.1, 0.3), "search": (0.3, 0.8)},
    },
    "hard": {
        "scale": (0.85, 1.15),
        "rotation_deg": (-5.0, 5.0),
        "blur_sigma": {"ref": (0.70, 1.30), "search": (1.30, 2.40)},
        "brightness": {"ref": (-32, 32), "search": (-38, 38)},
        "contrast": {"ref": (0.62, 1.35), "search": (0.58, 1.38)},
        "edge_gain": {"ref": (0.12, 0.65), "search": (0.10, 0.60)},
        "shading_amplitude": {"ref": (0.10, 0.22), "search": (0.12, 0.26)},
        "noise_sigma": {"ref": (4.0, 8.0), "search": (8.0, 16.0)},
        "dose": {"ref": (300, 900), "search": (40, 180)},
        "thermal_drift": {"ref": (0.30, 1.20), "search": (2.00, 5.00)},
        "vibration_amp": {"ref": (0.15, 0.60), "search": (0.60, 1.80)},
        "salt_pepper_prob": {"ref": (0.0010, 0.003), "search": (0.003, 0.008)},  # moderate impulse
        "impulse_burst_prob": {"ref": (0.0001, 0.0005), "search": (0.0005, 0.0015)},
        "brownian_sigma": {"ref": (0.2, 0.5), "search": (0.5, 1.2)},
    },
}

# Vibration frequency is expressed in cycles per frame height and is not tied to
# difficulty -- a mechanically noisy room is noisy for easy samples too.
VIBRATION_FREQ_RANGE = (3.0, 60.0)

# Distinct offsets per split guarantee that train / validation / test never draw
# from the same random stream (QC check 11).
SPLIT_SEED_OFFSET = {
    "train": 0,
    "validation": 1_000_003,
    "test": 2_000_003,
}


def _u(rng: np.random.Generator, lohi) -> float:
    """Uniform draw from a (low, high) tuple."""
    lo, hi = lohi
    return float(rng.uniform(lo, hi))


def _drift_axis(rng: np.random.Generator, lo: float, hi: float) -> float:
    """One axis of stage positioning error about the centre of the frame.

    A truncated Gaussian: draws beyond DRIFT_TRUNCATE_SIGMA are redrawn rather
    than clipped, because clipping would pile the whole tail onto the boundary
    and put a spike of samples at exactly one offset. `lo`/`hi` keep the
    ground-truth box inside the image; with sigma 110 px and a 3-sigma cut the
    bound is only reached for an unusually large footprint.
    """
    mid = SEARCH_SIZE_PX / 2.0
    limit = DRIFT_TRUNCATE_SIGMA * DRIFT_SIGMA_PX
    for _ in range(64):
        value = mid + float(rng.normal(0.0, DRIFT_SIGMA_PX))
        if abs(value - mid) <= limit and lo <= value <= hi:
            return value
    # Unreachable in practice; keeps the draw finite if the margins ever tighten.
    return float(min(max(mid, lo), hi))


def sample_seeds(sample_seed: int) -> dict:
    """Derive the independent sub-seeds for one sample.

    Every stochastic stage gets its own seed so that the reference noise and the
    search noise can never be the same array -- the requirement that the two
    images are independent physical captures is enforced structurally rather
    than by convention.
    """
    kids = np.random.SeedSequence(int(sample_seed)).generate_state(6, dtype=np.uint32)
    return {
        "param_seed": int(kids[0]),
        "geometry_seed": int(kids[1]),
        "reference_noise_seed": int(kids[2]),
        "search_noise_seed": int(kids[3]),
        "reference_scan_seed": int(kids[4]),
        "search_scan_seed": int(kids[5]),
    }


def make_sample_seed(master_seed: int, split: str, index: int) -> int:
    """Deterministic, globally unique per-sample seed."""
    base = int(master_seed) + SPLIT_SEED_OFFSET[split]
    return int(np.random.SeedSequence([base, int(index)]).generate_state(1, dtype=np.uint32)[0])


def assign_difficulty(index: int, n_total: int, rng: np.random.Generator) -> str:
    """Deterministic 25 / 50 / 25 split, shuffled so difficulty does not
    correlate with sample index."""
    n_easy = int(round(DIFFICULTY_MIX["easy"] * n_total))
    n_hard = int(round(DIFFICULTY_MIX["hard"] * n_total))
    n_medium = n_total - n_easy - n_hard
    labels = np.array(["easy"] * n_easy + ["medium"] * n_medium + ["hard"] * n_hard)
    rng.shuffle(labels)
    return str(labels[index])


def build_difficulty_plan(n_total: int, master_seed: int, split: str) -> list:
    """Pre-compute the difficulty label for every sample in a split."""
    rng = np.random.default_rng(int(master_seed) + SPLIT_SEED_OFFSET[split] + 77)
    n_easy = int(round(DIFFICULTY_MIX["easy"] * n_total))
    n_hard = int(round(DIFFICULTY_MIX["hard"] * n_total))
    n_medium = max(n_total - n_easy - n_hard, 0)
    labels = np.array(["easy"] * n_easy + ["medium"] * n_medium + ["hard"] * n_hard)
    if len(labels) < n_total:                      # rounding slack
        labels = np.concatenate([labels, np.array(["medium"] * (n_total - len(labels)))])
    labels = labels[:n_total]
    rng.shuffle(labels)
    return [str(v) for v in labels]


def _imaging_block(rng: np.random.Generator, tier: dict, side: str,
                   difficulty: str) -> dict:
    """Draw the per-image (reference or search) imaging / SEM / scan-error
    parameters. `side` is "ref" or "search"."""
    # P16 now comes from the randomized artifact stack, which redraws every
    # inner parameter of every noise model per image -- heavy for the search
    # capture, mediocre for the reference one.
    profile = draw_profile(rng, "reference" if side == "ref" else "search", difficulty)
    return {
        # P11
        "blur_sigma": _u(rng, tier["blur_sigma"][side]),
        # P12
        "brightness": _u(rng, tier["brightness"][side]),
        # P13
        "contrast": _u(rng, tier["contrast"][side]),
        # P14
        "edge_gain": _u(rng, tier["edge_gain"][side]),
        # P15
        "shading_amplitude": _u(rng, tier["shading_amplitude"][side]),
        # P16 -- surfaced at top level for convenience; the full stack is in
        # noise_profile, and that is what the pipeline actually applies.
        "noise_sigma": profile["detector_sigma"],
        "poisson_dose": profile["dose"],
        "noise_profile": profile,
        # Progressive raster shear, folded into the per-row scan displacement so
        # the ground truth follows it automatically.
        "raster_shear_px": profile["raster_shear_px"],
        # Radial scan-linearity error; ground truth is mapped through it
        # analytically in sample.build_sample.
        "barrel_k": profile["barrel_k"],
        # P19 - thermal drift: total displacement accumulated over one frame,
        # plus the direction it drifts in.
        "thermal_drift_px": _u(rng, tier["thermal_drift"][side]),
        "thermal_drift_angle_deg": float(rng.uniform(0.0, 360.0)),
        # P20 - vibration: quasi-periodic stage/column oscillation during the
        # raster scan, plus an incoherent row-to-row jitter component.
        "vibration_amp_px": _u(rng, tier["vibration_amp"][side]),
        "vibration_freq_cycles": _u(rng, VIBRATION_FREQ_RANGE),
        "vibration_phase_rad": float(rng.uniform(0.0, 2.0 * np.pi)),
        "vibration_jitter_px": _u(rng, tier["vibration_amp"][side]) * float(rng.uniform(0.0, 0.35)),
    }


def sample_parameters(sample_seed: int, difficulty: str) -> dict:
    """Draw the full P01-P20 parameter set for one sample.

    Everything is drawn from continuous distributions off a per-sample seed, so
    two samples sharing an identical parameter vector is a measure-zero event --
    this is what keeps the dataset from degenerating into one pattern repeated
    1000 times with different noise.
    """
    seeds = sample_seeds(sample_seed)
    rng = np.random.default_rng(seeds["param_seed"])
    tier = TIERS[difficulty]

    # ---- Level 1: architecture ------------------------------------------- #
    word_line_pitch = _u(rng, ARCH_RANGES["word_line_pitch"])                 # P01
    bit_ratio = _u(rng, ARCH_RANGES["bit_to_word_pitch_ratio"])               # P02
    line_width_frac = _u(rng, ARCH_RANGES["line_width_frac"])                 # P03
    contact_diameter_frac = _u(rng, ARCH_RANGES["contact_diameter_frac"])     # P04
    block_size = _u(rng, ARCH_RANGES["block_size"])
    block_line_count = int(max(round(block_size / word_line_pitch), 4))       # P05

    # ---- Level 2: geometry ------------------------------------------------ #
    spacing_jitter = _u(rng, GEOM_RANGES["spacing_jitter"])                   # P06
    phase_offset = [float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0))]  # P07

    defect_cfg = DEFECT_RANGES[difficulty]                                    # P08
    if rng.random() < defect_cfg["p_any"]:
        defect_density = _u(rng, defect_cfg["density"])
    else:
        defect_density = 0.0

    # ---- Level 3: imaging (shared geometry-affecting terms) --------------- #
    scale = _u(rng, tier["scale"])                                            # P09
    rotation_deg = _u(rng, tier["rotation_deg"])                              # P10

    # ---- Level 5: placement ----------------------------------------------- #
    footprint = NOMINAL_FOOTPRINT_PX * scale
    # Keep the rotated square, plus the worst-case scan displacement, strictly
    # inside the frame so the ground-truth box can never leave the image.
    # The allowance has to cover every geometry-moving effect at once: thermal
    # drift (<=5 px), vibration (<=1.8), raster shear (<=6) and row jitter.
    half_diag = 0.5 * footprint * (
        abs(np.cos(np.deg2rad(rotation_deg))) + abs(np.sin(np.deg2rad(rotation_deg)))
    )
    margin = float(half_diag + 26.0)
    lo, hi = margin, SEARCH_SIZE_PX - margin
    if rng.random() < DRIFT_FRACTION:
        position_mode = "drift"                                               # P17/P18
        position_x = _drift_axis(rng, lo, hi)
        position_y = _drift_axis(rng, lo, hi)
    else:
        position_mode = "uniform"
        position_x = float(rng.uniform(lo, hi))
        position_y = float(rng.uniform(lo, hi))

    reference_size_px = int(rng.integers(REFERENCE_SIZE_RANGE[0], REFERENCE_SIZE_RANGE[1] + 1))

    return {
        "difficulty": difficulty,
        "sample_seed": int(sample_seed),
        "seeds": seeds,

        # Level 1
        "word_line_pitch": word_line_pitch,
        "bit_to_word_pitch_ratio": bit_ratio,
        "bit_line_pitch": word_line_pitch * bit_ratio,
        "line_width_frac": line_width_frac,
        "contact_diameter_frac": contact_diameter_frac,
        "block_line_count": block_line_count,
        "block_size": block_size,
        "strip_width": _u(rng, ARCH_RANGES["strip_width"]),

        # Level 2
        "spacing_jitter": spacing_jitter,
        "phase_offset": phase_offset,
        "defect_density": defect_density,

        # Level 3 (geometry-affecting)
        "scale": scale,
        "rotation_deg": rotation_deg,
        "footprint_px": float(footprint),

        # Level 5
        "position_x": position_x,
        "position_y": position_y,
        "position_mode": position_mode,

        # Stored reference resolution
        "reference_size_px": reference_size_px,

        # Structural: gaps narrower than this bridge during processing. Scaled
        # to the lattice, and only active on samples that carry defects at all.
        "collapse_threshold": (
            float(rng.uniform(0.15, 0.40)) * word_line_pitch if defect_density > 0 else 0.0
        ),

        # Level 3/4 + P19/P20, drawn independently per capture
        "reference_imaging": _imaging_block(rng, tier, "ref", difficulty),
        "search_imaging": _imaging_block(rng, tier, "search", difficulty),
    }


def architecture_signature(params: dict) -> tuple:
    """A rounded fingerprint of a sample's Level 1+2 geometry, used by the
    leakage check to confirm that no geometric combination is reused."""
    return (
        round(params["word_line_pitch"], 4),
        round(params["bit_to_word_pitch_ratio"], 4),
        round(params["line_width_frac"], 4),
        round(params["contact_diameter_frac"], 4),
        params["block_line_count"],
        round(params["spacing_jitter"], 4),
        round(params["phase_offset"][0], 4),
        round(params["phase_offset"][1], 4),
    )
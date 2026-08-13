"""
Randomized SEM artifact stack.

The artifact primitives here are the ones from the original Drift-Sense
`sem_imaging.py` and `structural_defects.py` -- shot noise, detector noise,
speckle, salt-and-pepper, charging streaks, vignetting, gamma, barrel
distortion, raster drift and pattern collapse -- with their *inner* parameters
redrawn for every single image instead of being fixed constants. That is what
stops a thousand samples from sharing one noise character.

Two severity profiles are drawn per sample:

    search image     heavy       low dose, strong read noise, visible speckle
                                 and impulse noise, charging streaks, vignetting
    reference image  mediocre    the careful, slow, high-dose capture: real
                                 artifacts, but an order of magnitude gentler

Both are then scaled by the sample's difficulty tier.

Geometry safety
---------------
Most of these effects only change pixel *values*, so they can be applied freely.
Two of them move pixel *positions*, and applying those to the search image
without accounting for them would silently corrupt every ground-truth box:

  * `apply_barrel_distortion` -- handled by `barrel_forward_map_points`, which
    inverts the radial map analytically so annotations follow the pixels.
  * `apply_raster_drift` -- its progressive shear is folded into the per-row
    displacement in `sem.scan_displacement`, so the existing
    `sem.forward_map_points` already carries the ground truth through it.

Nothing in this module is applied to the search image without one of those two
paths keeping the coordinates exact.

(The module is named `random.py` as requested. It lives inside the `dram_synth`
package, not on `sys.path`, so it does not shadow the standard library's
`random` -- `import random` anywhere in this project still resolves to stdlib.)
"""

from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Value-only artifacts -- safe to apply anywhere
# --------------------------------------------------------------------------- #


def add_shot_noise(img: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise. `dose` proxies electron count / dwell time: a higher
    dose is a slower, more careful scan and therefore less relative noise."""
    if dose <= 0:
        return img
    counts = np.clip(img.astype(np.float64) / 255.0 * dose, 0, None)
    noisy = rng.poisson(counts).astype(np.float64) / dose * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_detector_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian read noise -- constant regardless of signal level."""
    if sigma <= 0:
        return img
    out = img.astype(np.float64) + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_speckle_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative noise, out = img * (1 + N(0, sigma)).

    Distinct from the additive read noise above: the magnitude scales with
    brightness, so bright features get noisier than dark background. Stands in
    for detector gain variation.
    """
    if sigma <= 0:
        return img
    out = img.astype(np.float64) * (1.0 + rng.normal(0.0, sigma, size=img.shape))
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(img: np.ndarray, prob: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Impulse noise: a fraction `prob` of pixels forced to 0 or 255.

    Structurally different from the smooth noise models -- dead and hot detector
    pixels, and sudden discharge events. Median-style filtering handles it,
    Gaussian denoising does not, which is exactly why it belongs in the mix.
    """
    if prob <= 0:
        return img
    out = img.copy()
    hit = rng.random(img.shape) < prob
    salt = rng.random(img.shape) < 0.5
    out[hit & salt] = 255
    out[hit & ~salt] = 0
    return out


def add_impulse_burst_noise(img: np.ndarray, burst_prob: float, burst_size: int = 3,
                             rng: np.random.Generator = None) -> np.ndarray:
    """Correlated impulse clusters (e.g., hot pixel clusters, discharge streaks).
    A fraction `burst_prob` of pixel locations spawn a burst of adjacent impulses.
    """
    if burst_prob <= 0 or rng is None:
        return img
    out = img.copy().astype(np.float64)
    h, w = img.shape
    
    # Random burst centers
    n_bursts = max(1, int(np.ceil(burst_prob * h * w / (burst_size ** 2))))
    if n_bursts > 0:
        centers_y = rng.integers(0, h, size=n_bursts)
        centers_x = rng.integers(0, w, size=n_bursts)
        
        for cy, cx in zip(centers_y, centers_x):
            # Random burst extent
            size = rng.integers(2, burst_size + 1)
            y_start = max(0, cy - size)
            y_end = min(h, cy + size + 1)
            x_start = max(0, cx - size)
            x_end = min(w, cx + size + 1)
            
            # Blast white or black
            if rng.random() < 0.5:
                out[y_start:y_end, x_start:x_end] = 255
            else:
                out[y_start:y_end, x_start:x_end] = 0
    
    return np.clip(out, 0, 255).astype(np.uint8)


def add_brownian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator = None) -> np.ndarray:
    """Low-frequency colored noise via cumulative summation (1/f-like).
    Simulates slow thermal drift of pixel values.
    """
    if sigma <= 0 or rng is None:
        return img
    
    h, w = img.shape
    # Generate white noise and integrate along each axis for low-frequency structure
    white_noise = rng.normal(0, sigma / 4.0, size=(h, w))
    brown_y = np.cumsum(white_noise, axis=0)
    brown_y = (brown_y - brown_y.min()) / (brown_y.max() - brown_y.min() + 1e-8) * (sigma * 2)
    brown_y -= sigma
    
    out = img.astype(np.float64) + brown_y
    return np.clip(out, 0, 255).astype(np.uint8)


def add_charging_streaks(img: np.ndarray, streak_prob: float, intensity: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Bright horizontal streaks from local charging on insulating regions.

    `streak_prob` is the expected number of streaks per 100 rows; `intensity`
    scales their brightness.
    """
    if streak_prob <= 0 or intensity <= 0:
        return img
    h, w = img.shape
    out = img.astype(np.float64)
    n_streaks = rng.poisson(max(streak_prob * (h / 100.0), 0))
    for _ in range(int(n_streaks)):
        row = int(rng.integers(0, h))
        band = max(1, int(rng.normal(2, 1)))
        lo, hi = max(row - band, 0), min(row + band, h)
        out[lo:hi, :] += intensity * rng.uniform(0.5, 1.0) * 25.5
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """Radial darkening toward the frame edges, from off-axis collection
    efficiency falloff. `strength` in [0, 1]."""
    if strength <= 0:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    r = np.clip(r / np.sqrt(2), 0, 1)
    out = img.astype(np.float64) * (1.0 - strength * r ** 2)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Nonlinear detector response / mis-set contrast curve. 1.0 is a no-op."""
    if gamma == 1.0:
        return img
    norm = np.clip(img.astype(np.float64) / 255.0, 0, 1)
    return np.clip(np.power(norm, gamma) * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Geometry-moving artifacts -- each paired with its coordinate map
# --------------------------------------------------------------------------- #


def apply_barrel_distortion(img: np.ndarray, k: float) -> np.ndarray:
    """Radial scan-linearity error: barrel for k > 0, pincushion for k < 0.

    Sampling is out(x, y) = in(distort(x, y)), where distort scales the
    normalized radius by (1 + k*r^2).
    """
    if k == 0.0:
        return img
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    factor = 1.0 + k * (nx ** 2 + ny ** 2)
    map_x = (nx * factor) * cx + cx
    map_y = (ny * factor) * cy + cy
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def barrel_forward_map_points(points: np.ndarray, k: float,
                              width: int, height: int) -> np.ndarray:
    """Where do specimen points land after `apply_barrel_distortion`?

    The warp samples the source at radius R = r*(1 + k*r^2), so a feature at
    source radius R appears at output radius r solving r + k*r^3 = R. That cubic
    is solved by Newton iteration from r = R; it converges in a handful of steps
    for the small |k| used here. Direction from the centre is preserved, so only
    the radius changes.

    This is what keeps the ground truth exact when barrel distortion is applied
    to the search image, instead of the annotation quietly drifting off the
    feature it names.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if k == 0.0:
        return pts.copy()

    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    nx = (pts[:, 0] - cx) / cx
    ny = (pts[:, 1] - cy) / cy
    big_r = np.hypot(nx, ny)

    r = big_r.copy()
    for _ in range(12):
        f = r + k * r ** 3 - big_r
        df = 1.0 + 3.0 * k * r ** 2
        step = f / np.where(np.abs(df) < 1e-12, 1e-12, df)
        r = r - step
        if np.max(np.abs(step)) < 1e-12:
            break

    scale = np.where(big_r > 1e-12, r / np.where(big_r > 1e-12, big_r, 1.0), 1.0)
    out = np.empty_like(pts)
    out[:, 0] = nx * scale * cx + cx
    out[:, 1] = ny * scale * cy + cy
    return out


def apply_raster_drift(img: np.ndarray, shear_amplitude_px: float,
                       jitter_std_px: float, rng: np.random.Generator) -> np.ndarray:
    """Progressive row-to-row shear plus per-row jitter.

    Kept here in its original standalone form. In the generation pipeline the
    shear term is instead folded into `sem.scan_displacement`, so that
    `sem.forward_map_points` carries the ground truth through it automatically;
    this function is for standalone use and for augmenting an image whose
    coordinates do not need tracking.
    """
    if shear_amplitude_px == 0 and jitter_std_px == 0:
        return img
    h, w = img.shape
    shear = shear_amplitude_px * (np.arange(h) / max(h - 1, 1))
    jitter = rng.normal(0, jitter_std_px, size=h) if jitter_std_px > 0 else np.zeros(h)
    row_shift = (shear + jitter).astype(np.float32)
    map_x = np.arange(w, dtype=np.float32)[None, :] + row_shift[:, None]
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------- #
# Structural defect (a property of the device, not of the capture)
# --------------------------------------------------------------------------- #


def maybe_collapse_gap(gap: float, threshold: float, rng: np.random.Generator,
                       collapse_prob: float = 0.7) -> bool:
    """Should the gap between two adjacent lines bridge?

    High-aspect-ratio lines topple and stick to their neighbour under capillary
    forces during processing. Gaps at or above `threshold` never collapse; below
    it they collapse with `collapse_prob`, so the effect is visible without being
    deterministic. Applied to the specimen, so a bridged pair appears in *both*
    the reference and the search image -- unlike everything else in this module,
    which is per-capture.
    """
    if gap >= threshold:
        return False
    return bool(rng.random() < collapse_prob)


# --------------------------------------------------------------------------- #
# Severity profiles
# --------------------------------------------------------------------------- #

# Base ranges per capture. The search is the fast, wide-area scan and is
# punished accordingly; the reference is the careful high-dose capture.
_BASE = {
    "reference": {
        "dose": (400.0, 1600.0),            # high dose -> mild shot noise
        "detector_sigma": (1.5, 5.0),
        "speckle_sigma": (0.02, 0.10),
        "salt_pepper_prob": (0.0000, 0.0010),
        "impulse_burst_prob": (0.0000, 0.0001),
        "brownian_sigma": (0.0, 0.3),
        "charging_streak_prob": (0.0, 0.9),
        "charging_streak_intensity": (0.0, 0.8),
        "vignette_strength": (0.02, 0.12),
        "gamma": (0.90, 1.15),
        # Either sign is fine here: distorting the reference image does not move
        # anything the ground truth refers to.
        "barrel_k": (-0.045, 0.045),
        "raster_shear_px": (0.0, 0.6),
    },
    "search": {
        "dose": (25.0, 260.0),              # low dose -> heavy shot noise
        "detector_sigma": (6.0, 18.0),
        "speckle_sigma": (0.10, 0.38),
        "salt_pepper_prob": (0.0010, 0.0120),
        "impulse_burst_prob": (0.0002, 0.0040),
        "brownian_sigma": (0.2, 1.5),
        "charging_streak_prob": (0.5, 3.8),
        "charging_streak_intensity": (0.4, 2.4),
        "vignette_strength": (0.08, 0.32),
        "gamma": (0.72, 1.38),
        # Barrel only (k >= 0) on the search side. Barrel moves content *toward*
        # the frame centre, so a ground-truth box can never be pushed out of the
        # image; pincushion (k < 0) moves it outward and near the edge that
        # growth is large enough to carry a box off-frame. The reference gets
        # both signs, where it costs nothing.
        #
        # Kept mild. Barrel is a *non-affine* warp: it shrinks the reference
        # region locally, and by different amounts radially and tangentially. At
        # k ~ 0.07 near the frame edge that is a ~9% size change, which stops
        # being a plausible scan-linearity error and starts being a different
        # image. At 0.03 it is a few percent -- visible, realistic, matchable.
        "barrel_k": (0.0, 0.030),
        "raster_shear_px": (0.4, 4.5),
    },
}

# Difficulty scales severity. Dose moves the other way -- less dose is worse.
_SEVERITY = {"easy": 0.55, "medium": 1.00, "hard": 1.50}

# Parameters measured as a deviation from 1.0 rather than from 0.
_CENTERED_AT_ONE = ("gamma",)
# Parameters where a larger value is *better*.
_INVERSE = ("dose",)
# Hard ceilings, so a hard-tier draw cannot produce a physically silly image.
_CLAMP = {
    "speckle_sigma": (0.0, 0.55),
    "salt_pepper_prob": (0.0, 0.025),
    "impulse_burst_prob": (0.0, 0.008),
    "brownian_sigma": (0.0, 3.0),
    "vignette_strength": (0.0, 0.45),
    "gamma": (0.55, 1.60),
    "barrel_k": (-0.050, 0.050),
    "dose": (18.0, 4000.0),
    "detector_sigma": (0.0, 26.0),
    "raster_shear_px": (0.0, 6.0),
    "charging_streak_prob": (0.0, 5.0),
    "charging_streak_intensity": (0.0, 3.0),
}


def draw_profile(rng: np.random.Generator, side: str, difficulty: str) -> dict:
    """Draw a full set of artifact parameters for one capture.

    `side` is "reference" or "search". Every parameter is an independent draw,
    so no two images in the dataset share a noise character.
    """
    severity = _SEVERITY[difficulty]
    profile = {}
    for name, (lo, hi) in _BASE[side].items():
        value = float(rng.uniform(lo, hi))
        if name in _INVERSE:
            value /= severity
        elif name in _CENTERED_AT_ONE:
            value = 1.0 + (value - 1.0) * severity
        else:
            value *= severity
        if name in _CLAMP:
            c_lo, c_hi = _CLAMP[name]
            value = float(np.clip(value, c_lo, c_hi))
        profile[name] = value

    # Not every capture shows every artifact. Switching some off at random keeps
    # the dataset from having one uniform "look" that a model could rely on.
    if rng.random() < (0.45 if side == "reference" else 0.20):
        profile["charging_streak_prob"] = 0.0
    if rng.random() < (0.55 if side == "reference" else 0.25):
        profile["salt_pepper_prob"] = 0.0
    if rng.random() < (0.65 if side == "reference" else 0.30):
        profile["impulse_burst_prob"] = 0.0
    if rng.random() < (0.50 if side == "reference" else 0.35):
        profile["brownian_sigma"] = 0.0
    if rng.random() < 0.35:
        profile["barrel_k"] = 0.0

    profile["side"] = side
    return profile


def apply_profile(img: np.ndarray, profile: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply the value-only part of a profile, in physical order.

    Signal degradation first (shot noise is generated at the specimen, read
    noise at the detector), then the response curve, then charging -- which is a
    beam-sample interaction and so is not attenuated by the detector's vignette.

    The geometry-moving members of the profile (`barrel_k`, `raster_shear_px`)
    are deliberately *not* applied here; they are applied in `sem.image_capture`
    where their coordinate maps are available.
    """
    img = add_shot_noise(img, profile["dose"], rng)
    img = add_detector_noise(img, profile["detector_sigma"], rng)
    img = add_speckle_noise(img, profile["speckle_sigma"], rng)
    img = add_salt_and_pepper_noise(img, profile["salt_pepper_prob"], rng)
    img = add_impulse_burst_noise(img, profile["impulse_burst_prob"], 
                                  burst_size=4, rng=rng)
    img = add_brownian_noise(img, profile["brownian_sigma"], rng=rng)
    img = apply_vignette(img, profile["vignette_strength"])
    img = apply_gamma(img, profile["gamma"])
    img = add_charging_streaks(img, profile["charging_streak_prob"],
                               profile["charging_streak_intensity"], rng)
    return img

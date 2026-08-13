"""
The SEM imaging chain.

Everything here models *how the specimen was captured*, not what it is. The
reference and the search image run through this chain separately, with
independently drawn parameters and independently seeded random generators, so
they behave like two genuinely separate acquisitions of the same device.

Chain order (per image):

    1. edge brightening      secondary-electron yield rises at feature edges
    2. downsample            detector sampling of the supersampled specimen
    3. thermal drift         slow monotonic displacement over the frame  (P19)
    4. vibration             quasi-periodic row displacement + jitter    (P20)
       raster shear          progressive sideways walk of the scan
    5. barrel distortion     radial scan-linearity error
    6. optical blur          beam spot / defocus                         (P11)
    7. shading               low-frequency collection-efficiency falloff (P15)
    8. contrast / brightness detector gain and offset                (P13/P12)
    9. artifact stack        shot, read, speckle and impulse noise, vignetting,
                             gamma and charging streaks -- every inner parameter
                             redrawn per image, from `random.py`          (P16)

Steps 3-5 move image content, so the ground-truth coordinates are pushed through
the *same* analytic maps rather than being re-measured from the warped image:
`scan_displacement` / `forward_map_points` for the row displacements, and
`random.barrel_forward_map_points` for the radial distortion.
"""

from __future__ import annotations

import cv2
import numpy as np

# Aliased on import: the module is named `random.py` but is not the standard
# library's, and `artifacts` says what it actually holds.
from . import random as artifacts


# --------------------------------------------------------------------------- #
# 1. SEM edge brightening
# --------------------------------------------------------------------------- #

def edge_brighten(img: np.ndarray, gain: float) -> np.ndarray:
    """Brighten feature boundaries.

    This is not a sharpening filter. The edge map is computed geometrically --
    a morphological gradient marks where material boundaries actually are -- and
    that map is then softened and *added* as extra emitted signal. The physical
    story is that more secondary electrons escape near an edge, so edges read
    brighter than the flat top of the same feature. Interiors are untouched.
    """
    if gain <= 0:
        return img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Buffers are reused via `dst=` throughout: at supersample 10 the specimen
    # is a 10000x10000 array, and each spare temporary costs another 100 MB.
    buf = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    cv2.GaussianBlur(buf, (0, 0), sigmaX=1.0, dst=buf)
    cv2.addWeighted(img, 1.0, buf, float(gain), 0.0, dst=buf)
    return buf


# --------------------------------------------------------------------------- #
# 2. Detector sampling
# --------------------------------------------------------------------------- #

def downsample(img: np.ndarray, out_size: int) -> np.ndarray:
    """Area-average down to the detector's pixel grid."""
    if img.shape[0] == out_size and img.shape[1] == out_size:
        return img
    return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# 3 + 4. Thermal drift and vibration
# --------------------------------------------------------------------------- #

def scan_displacement(height: int, imaging: dict, seed: int) -> tuple:
    """Per-row source displacement (dx, dy) caused by thermal drift and vibration.

    A raster scan builds an image one row at a time, so anything that moves the
    stage or the beam during the frame shows up as a row-dependent displacement:

      * thermal drift (P19) -- the stage expands as it warms, so the
        displacement accumulates monotonically over the frame. Modelled as
        0.75*t + 0.25*t^2 of the total drift, in a random direction: mostly
        linear with a slight acceleration, which is what a warming stage does.

      * vibration (P20) -- floor/pump/cooling-line coupling oscillates the
        column at a fixed frequency, giving a sinusoidal row displacement, plus
        an incoherent row-to-row jitter term for broadband mechanical noise.

    Returns (dx, dy), each of length `height`, in output-pixel units. They are
    *source* offsets: out[r, c] = in[r + dy[r], c + dx[r]].
    """
    rng = np.random.default_rng(int(seed))
    rows = np.arange(height, dtype=np.float64)
    t = rows / max(height - 1, 1)                      # normalized scan time

    drift = float(imaging["thermal_drift_px"])
    angle = np.deg2rad(float(imaging["thermal_drift_angle_deg"]))
    ramp = 0.75 * t + 0.25 * t ** 2
    dx = drift * ramp * np.cos(angle)
    dy = drift * ramp * np.sin(angle)

    # Progressive raster shear: the scan itself walks sideways as the frame is
    # built. Folded in here rather than applied as a separate warp, so that
    # `forward_map_points` carries the ground truth through it for free.
    dx = dx + float(imaging.get("raster_shear_px", 0.0)) * t

    amp = float(imaging["vibration_amp_px"])
    freq = float(imaging["vibration_freq_cycles"])
    phase = float(imaging["vibration_phase_rad"])
    dx = dx + amp * np.sin(2.0 * np.pi * freq * t + phase)
    # The vertical component of a vibration is weaker and at a different beat,
    # since the slow-scan axis integrates over a whole line time.
    dy = dy + 0.35 * amp * np.sin(np.pi * freq * t + 1.7 * phase)

    jitter = float(imaging["vibration_jitter_px"])
    if jitter > 0:
        dx = dx + rng.normal(0.0, jitter, size=height)
        dy = dy + rng.normal(0.0, 0.4 * jitter, size=height)

    return dx, dy


def apply_scan_displacement(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Warp an image by the per-row displacement from `scan_displacement`."""
    if not np.any(dx) and not np.any(dy):
        return img
    h, w = img.shape
    map_x = np.arange(w, dtype=np.float32)[None, :] + dx[:, None].astype(np.float32)
    map_y = np.arange(h, dtype=np.float32)[:, None] + dy[:, None].astype(np.float32)
    map_y = np.broadcast_to(map_y, (h, w)).copy()
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def forward_map_points(points: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Map specimen points to where they land in the distorted image.

    `apply_scan_displacement` samples the source at (r + dy[r], c + dx[r]), so a
    feature at source (X, Y) appears at the output row r solving r + dy[r] = Y,
    and at column X - dx[r]. dy is smooth and small, so three fixed-point
    iterations converge to well below a hundredth of a pixel.

    This is the inverse of measuring the ground truth from the warped image --
    the transformation is tracked mathematically, so the annotation stays exact
    no matter how strong the drift and vibration are.
    """
    h = len(dx)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(pts)
    for i, (x, y) in enumerate(pts):
        r = y
        for _ in range(3):
            ri = int(np.clip(round(r), 0, h - 1))
            r = y - dy[ri]
        ri = int(np.clip(round(r), 0, h - 1))
        out[i] = (x - dx[ri], r)
    return out


# --------------------------------------------------------------------------- #
# 5 - 8. Optics, shading, gain, noise
# --------------------------------------------------------------------------- #

def optical_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian beam-spot / defocus blur."""
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


def apply_shading(img: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth multiplicative intensity variation across the frame.

    Off-axis detector collection efficiency and specimen tilt make the signal
    level drift slowly across a frame. Built by upsampling a small random field,
    so it is genuinely low-frequency and spatially varying rather than a fixed
    radial vignette.
    """
    if amplitude <= 0:
        return img
    h, w = img.shape
    coarse = rng.normal(0.0, 1.0, size=(5, 5)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    field = field / (np.abs(field).max() + 1e-6)
    out = img.astype(np.float32) * (1.0 + amplitude * field)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gain(img: np.ndarray, contrast: float, brightness: float) -> np.ndarray:
    """Detector contrast (gain) and brightness (offset), about mid-grey."""
    out = (img.astype(np.float32) - 128.0) * float(contrast) + 128.0 + float(brightness)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_sensor_noise(img: np.ndarray, dose: float, sigma: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise plus Gaussian read noise, from a caller-owned generator.

    The generator is supplied by the caller and is seeded per image, which is how
    reference and search noise are kept independent -- there is no shared noise
    array anywhere in this module.
    """
    out = img.astype(np.float64)
    if dose > 0:
        counts = np.clip(out / 255.0 * dose, 0, None)
        out = rng.poisson(counts).astype(np.float64) / dose * 255.0
    if sigma > 0:
        out = out + rng.normal(0.0, float(sigma), size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def local_blur_defect(img: np.ndarray, rng: np.random.Generator) -> tuple:
    """A locally out-of-focus patch (specimen tilt / local charging).

    Returns (image, region) where region is None if nothing was applied.
    """
    h, w = img.shape
    side = int(rng.uniform(0.06, 0.18) * min(h, w))
    y0 = int(rng.integers(0, max(h - side, 1)))
    x0 = int(rng.integers(0, max(w - side, 1)))
    sigma = float(rng.uniform(1.2, 3.5))
    patch = img[y0:y0 + side, x0:x0 + side]
    img = img.copy()
    img[y0:y0 + side, x0:x0 + side] = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma)
    return img, {"x": x0, "y": y0, "size": side, "sigma": round(sigma, 3)}


def image_capture(specimen: np.ndarray, out_size: int, imaging: dict,
                  noise_seed: int, scan_seed: int,
                  local_blur: bool = False) -> tuple:
    """Run the full capture chain on a specimen image.

    Returns (image, dx, dy, extra) where dx/dy are the per-row scan
    displacements used, so the caller can push ground-truth coordinates through
    exactly the same transformation.
    """
    rng = np.random.default_rng(int(noise_seed))

    img = edge_brighten(specimen, imaging["edge_gain"])
    img = downsample(img, out_size)

    dx, dy = scan_displacement(out_size, imaging, scan_seed)
    img = apply_scan_displacement(img, dx, dy)

    # Radial scan-linearity error. This moves content, so the caller must map
    # ground-truth points through `artifacts.barrel_forward_map_points` with the
    # same k -- which `sample.build_sample` does.
    barrel_k = float(imaging.get("barrel_k", 0.0))
    img = artifacts.apply_barrel_distortion(img, barrel_k)

    img = optical_blur(img, imaging["blur_sigma"])
    img = apply_shading(img, imaging["shading_amplitude"], rng)
    img = apply_gain(img, imaging["contrast"], imaging["brightness"])

    # The randomized artifact stack: shot noise, read noise, speckle, impulse
    # noise, vignetting, gamma and charging streaks, every inner parameter drawn
    # fresh for this image. Falls back to the plain two-term noise model if a
    # caller supplies an imaging block without a profile.
    profile = imaging.get("noise_profile")
    if profile is not None:
        img = artifacts.apply_profile(img, profile, rng)
    else:
        img = add_sensor_noise(img, imaging["poisson_dose"], imaging["noise_sigma"], rng)

    extra = {}
    if local_blur:
        img, region = local_blur_defect(img, rng)
        extra["local_blur"] = region

    return img, dx, dy, extra
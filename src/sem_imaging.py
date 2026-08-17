"""
SEM acquisition artifacts -- applied per-image, since the reference and
search captures happen under different conditions (careful/slow vs.
fast/wide-area). This is deliberately kept separate from
structural_defects.py, which models a property of the physical device
rather than of how it was imaged.

There is a single physical beam (`beam_spot_size_nm`), applied identically
to both images as a Gaussian PSF blur *before* any downsampling. The search
image's extra softness on dense structures comes naturally from the 10x
area-average downsample on top of that shared blur -- not from a separate
"search-only blur" fudge factor.

Artifact primitives are *not* defined here. They live in the canonical engine
at `dram_dataset/dram_synth/random.py` and are imported below. The two copies
were verified pixel-for-pixel identical across all eleven shared functions
before being collapsed into one, so nothing about this module's output changed
when the duplicate definitions were removed. What remains here is only what is
genuinely specific to this pipeline's nm-calibrated framing: the beam PSF, the
10x area-average downsample, bit-depth quantization, and the two capture
recipes that sequence the primitives.
"""

import cv2
import numpy as np

from dram_dataset.dram_synth.random import (  # noqa: F401  (re-exported)
    add_brownian_noise,
    add_charging_streaks,
    add_detector_noise,
    add_impulse_burst_noise,
    add_salt_and_pepper_noise,
    add_shot_noise,
    add_speckle_noise,
    apply_barrel_distortion,
    apply_gamma,
    apply_raster_drift,
    apply_vignette,
)


def gaussian_psf_blur(
    img: np.ndarray,
    spot_size_nm: float,
    pixel_size_nm: float,
    astigmatism_ratio: float = 1.0,
) -> np.ndarray:
    """Gaussian beam-spot blur. `astigmatism_ratio` != 1.0 makes the spot
    elliptical (sigmaY = sigmaX * ratio) -- a real, common SEM aberration
    where the beam isn't perfectly round, which shows up as directional
    blurring (sharper along one scan axis than the other).
    """
    sigma_x = max(spot_size_nm / pixel_size_nm, 1e-6)
    sigma_y = max(sigma_x * astigmatism_ratio, 1e-6)
    k = int(2 * round(3 * max(sigma_x, sigma_y)) + 1)
    k = max(k, 3)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma_x, sigmaY=sigma_y)


def downsample_area_average(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def add_quantization_noise(img: np.ndarray, bits: int = 8, rng: np.random.Generator = None) -> np.ndarray:
    """Simulates reduced bit-depth (quantization + dithering).
    Reduces precision to `bits` levels; adds dithering for smoothness.
    """
    if bits >= 8:
        return img
    img_f = img.astype(np.float64)
    levels = (1 << bits) - 1  # 2^bits - 1
    step = 255.0 / levels
    quantized = np.round(img_f / step) * step
    if rng is not None:
        dither = rng.uniform(-step * 0.5, step * 0.5, size=img.shape)
        quantized = quantized + dither
    return np.clip(quantized, 0, 255).astype(np.uint8)


def image_reference(
    crop: np.ndarray,
    pixel_size_nm: float,
    spot_size_nm: float,
    dose: float,
    rng: np.random.Generator,
    detector_noise_sigma: float = 2.0,
    drift_jitter_px: float = 0.2,
    astigmatism_ratio: float = 1.0,
    vignette_strength: float = 0.0,
    gamma: float = 1.0,
    barrel_distortion_k: float = 0.0,
    charging_streak_prob: float = 0.0,
    charging_streak_intensity: float = 0.0,
    speckle_sigma: float = 0.0,
    salt_pepper_prob: float = 0.0,
) -> np.ndarray:
    img = gaussian_psf_blur(crop, spot_size_nm, pixel_size_nm, astigmatism_ratio)
    img = apply_raster_drift(img, shear_amplitude_px=0.0, jitter_std_px=drift_jitter_px, rng=rng)
    img = apply_barrel_distortion(img, barrel_distortion_k)
    img = add_shot_noise(img, dose, rng)
    img = add_detector_noise(img, detector_noise_sigma, rng)
    img = add_speckle_noise(img, speckle_sigma, rng)
    img = add_salt_and_pepper_noise(img, salt_pepper_prob, rng)
    img = apply_vignette(img, vignette_strength)
    img = apply_gamma(img, gamma)
    img = add_charging_streaks(img, charging_streak_prob, charging_streak_intensity, rng)
    return img


def image_search(
    full_canvas: np.ndarray,
    pixel_size_ref_nm: float,
    pixel_size_search_nm: float,
    spot_size_nm: float,
    dose: float,
    rng: np.random.Generator,
    shear_amplitude_px: float = 1.5,
    drift_jitter_px: float = 0.5,
    detector_noise_sigma: float = 5.0,
    astigmatism_ratio: float = 1.0,
    vignette_strength: float = 0.0,
    gamma: float = 1.0,
    barrel_distortion_k: float = 0.0,
    charging_streak_prob: float = 0.0,
    charging_streak_intensity: float = 0.0,
    speckle_sigma: float = 0.0,
    salt_pepper_prob: float = 0.0,
) -> np.ndarray:
    factor = int(round(pixel_size_search_nm / pixel_size_ref_nm))
    blurred = gaussian_psf_blur(full_canvas, spot_size_nm, pixel_size_ref_nm, astigmatism_ratio)
    downsampled = downsample_area_average(blurred, factor)
    drifted = apply_raster_drift(downsampled, shear_amplitude_px, drift_jitter_px, rng)
    distorted = apply_barrel_distortion(drifted, barrel_distortion_k)
    noisy = add_shot_noise(distorted, dose, rng)
    noisy = add_detector_noise(noisy, detector_noise_sigma, rng)
    noisy = add_speckle_noise(noisy, speckle_sigma, rng)
    noisy = add_salt_and_pepper_noise(noisy, salt_pepper_prob, rng)
    noisy = apply_vignette(noisy, vignette_strength)
    noisy = apply_gamma(noisy, gamma)
    noisy = add_charging_streaks(noisy, charging_streak_prob, charging_streak_intensity, rng)
    return noisy

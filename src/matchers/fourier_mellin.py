"""Fourier-Mellin. The geometry specialist.

Chain of facts it exploits:
  * a *translation* changes only the phase of the Fourier transform, so the
    magnitude spectrum is translation-invariant;
  * in that magnitude spectrum a rotation of the image is a rotation of the
    spectrum, and a scaling is an inverse scaling of the spectrum;
  * on a log-polar grid, rotation becomes a shift along the angle axis and
    scaling becomes a shift along the log-radius axis.

So: FFT -> magnitude -> high-pass -> log-polar -> phase correlation turns
"how is this rotated and scaled?" into "what is the translation between
these two images?", which we already know how to answer.

Cost is why this is *not* run everywhere. It is invoked on a candidate patch
once there is evidence of geometric mismatch -- a candidate that a plain
same-orientation correlation cannot explain -- and its output is a corrected
template that the final ZNCC stage can actually match.

Caveat kept deliberately visible: the magnitude spectrum of a real image is
centrosymmetric, so the recovered angle is ambiguous by 180 degrees. Both
hypotheses are scored and the better one wins.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .common import (
    DEFAULT_ANGLES,
    DEFAULT_SIZES,
    INNER_FRACTION,
    MatchResult,
    crop_roi,
    make_template,
)
from .wavelet import wavelet_candidates
from .zncc import zncc_refine

FM_SIZE = 128           # both images are resampled to this before the transform
FM_ANGLE_LIMIT = 30.0   # beyond this, treat the estimate as unreliable
FM_SCALE_LIMITS = (0.65, 1.55)
# On a log-radius axis the first few pixels of the spectrum -- the DC blob --
# occupy a third of the width and are identical whatever the rotation. Left in,
# they pin the phase correlation at zero shift; this is where we start reading.
FM_MIN_RADIUS = 3.0


def _highpass(shape: tuple) -> np.ndarray:
    """Reddy-Chatterji high-pass emphasis, H = (1-X)(2-X).

    The magnitude spectrum is dominated by its DC neighbourhood, which
    carries no orientation information and would otherwise drown the
    log-polar correlation.
    """
    h, w = shape
    y = np.linspace(-0.5, 0.5, h, dtype=np.float32).reshape(-1, 1)
    x = np.linspace(-0.5, 0.5, w, dtype=np.float32).reshape(1, -1)
    cross = np.cos(np.pi * y) * np.cos(np.pi * x)
    return (1.0 - cross) * (2.0 - cross)


def _log_polar_spectrum(img: np.ndarray, size: int = FM_SIZE) -> tuple:
    """|FFT| -> high-pass -> log-polar. Returns (log_polar, k_log)."""
    f = cv2.resize(img.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    f = f - f.mean()
    f *= np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)

    mag = np.abs(np.fft.fftshift(np.fft.fft2(f))).astype(np.float32)
    mag *= _highpass(mag.shape)
    mag = np.log1p(mag)

    center = (size / 2.0, size / 2.0)
    max_radius = size / 2.0
    lp = cv2.warpPolar(mag, (size, size), center, max_radius,
                       cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG)
    k_log = size / np.log(max_radius)

    # Drop the DC neighbourhood and window the (non-circular) radius axis. The
    # angle axis is genuinely circular, so it is deliberately left unwindowed.
    col0 = int(k_log * np.log(FM_MIN_RADIUS))
    lp = lp[:, col0:]
    lp = lp - lp.mean()
    lp *= np.hanning(lp.shape[1]).astype(np.float32).reshape(1, -1)
    return np.ascontiguousarray(lp, dtype=np.float64), k_log


def estimate_rotation_scale(template: np.ndarray, patch: np.ndarray,
                            size: int = FM_SIZE) -> dict:
    """Rotation (degrees) and scale taking `template` onto `patch`.

    Returns {"angle", "scale", "confidence"}. `confidence` is the phase
    correlation response in log-polar space; low values mean the two images
    do not differ by a similarity transform at all (different content), and
    the caller should not trust the numbers.
    """
    lp_t, k_log = _log_polar_spectrum(template, size)
    lp_p, _ = _log_polar_spectrum(patch, size)

    (dx, dy), response = cv2.phaseCorrelate(lp_t, lp_p)

    # cv2.phaseCorrelate reports (centre - peak), so the shift that carries the
    # template's spectrum onto the patch's is the negative of what it returns.
    # Rows of the log-polar image span 0..360 degrees; columns are log-radius.
    angle = -float(dy) * 360.0 / float(size)
    scale = float(np.exp(-float(dx) / k_log))

    # wrap into (-180, 180]
    angle = (angle + 180.0) % 360.0 - 180.0
    return {"angle": angle, "scale": scale, "confidence": float(response)}


def fourier_mellin_match(reference: np.ndarray, search: np.ndarray,
                         levels: int = 2, top_k: int = 5,
                         base_size: int = 100,
                         sizes=DEFAULT_SIZES,
                         method_name: str = "fourier_mellin") -> MatchResult:
    """Coarse candidates -> per-candidate geometry estimate -> ZNCC verify.

    Fourier-Mellin needs something to compare *against*, so a cheap wavelet
    stage proposes candidate patches first. Each candidate gets its own
    rotation/scale estimate, the template is corrected accordingly, and only
    then is ZNCC allowed to score it. That ordering is the whole point: ZNCC
    cannot recognize a patch it is misaligned with.
    """
    t0 = time.perf_counter()
    coarse = wavelet_candidates(reference, search, levels=levels, top_k=top_k)
    base_template = make_template(reference, int(base_size), 0.0, INNER_FRACTION)
    half = int(base_template.shape[0] * 0.85)

    best = None
    best_geom = None
    for cx, cy, _ in coarse["candidates"]:
        patch, _, _ = crop_roi(search, cx, cy, half)
        if min(patch.shape[:2]) < 24:
            continue
        geom = estimate_rotation_scale(base_template, patch)

        angle = geom["angle"]
        scale = geom["scale"]
        usable = (abs(angle) <= FM_ANGLE_LIMIT
                  and FM_SCALE_LIMITS[0] <= scale <= FM_SCALE_LIMITS[1])

        # The magnitude spectrum is centrosymmetric: angle and angle+180 are
        # indistinguishable here, so both are put to the ZNCC vote -- along
        # with the null hypothesis, so a bad estimate can never score worse
        # than not correcting at all.
        angle_hypotheses = [0.0]
        if usable:
            angle_hypotheses += [angle, angle + 180.0 if angle < 0 else angle - 180.0]
        corrected_size = int(round(base_size * scale)) if usable else int(base_size)
        corrected_size = int(np.clip(corrected_size, min(sizes), max(sizes)))
        size_sweep = (corrected_size, )

        for hyp in angle_hypotheses:
            refined = zncc_refine(reference, search, cx, cy, search_radius=12,
                                  sizes=size_sweep, angles=(-1.5, 0.0, 1.5),
                                  base_angle=float(hyp))
            if best is None or refined.score > best.score:
                best = refined
                best_geom = {"fm_angle": angle, "fm_scale": scale,
                             "fm_confidence": geom["confidence"],
                             "hypothesis_angle": float(hyp), "usable": usable}

    if best is None:
        raise ValueError("fourier-mellin: no usable candidate patch")

    best.method = method_name
    best.n_candidates = coarse["n_candidates"]
    best.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    best.extra.update(best_geom or {})
    best.extra["native_metric"] = "fm_confidence"
    best.extra["native_score"] = (best_geom or {}).get("fm_confidence", 0.0)
    return best

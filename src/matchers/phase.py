"""Phase correlation. The fast translation estimator.

Cross-power spectrum: R = F1 * conj(F2) / |F1 * conj(F2)|. Discarding the
magnitude and keeping only the phase makes the response a near-delta at the
displacement, which is why phase correlation is both fast (two FFTs) and
insensitive to a global brightness or contrast change.

Its assumptions are strict, and worth stating because they are exactly the
conditions this dataset violates on purpose:
  * it estimates *translation* only -- rotation or scale mismatch smears the
    delta into nothing;
  * whitening the spectrum amplifies noise as much as signal;
  * on a periodic scene the delta splits into a lattice of equal peaks.

So it belongs in the comparison as the cheap baseline that shows what pure
displacement estimation can and cannot do here, not as the final answer.
"""

from __future__ import annotations

import time

import numpy as np

from .common import (
    DEFAULT_SIZES,
    INNER_FRACTION,
    MatchResult,
    make_template,
    peak_to_sidelobe_ratio,
    subpixel_peak,
    zncc_score_at,
)

PHASE_SIZES = (85, 95, 105, 115)


def _hann2d(k: int) -> np.ndarray:
    w = np.hanning(k).astype(np.float32)
    return np.outer(w, w)


def phase_correlation_surface(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Cross-power-spectrum surface of a small template against a big image.

    The template is zero-meaned and Hann-windowed, then embedded at the origin
    of a search-sized zero array -- zero *is* the template's mean after
    centring, so the padding contributes nothing to the correlation. The peak
    at (u, v) is then the template's top-left corner in the search image.
    """
    h, w = search.shape
    k = template.shape[0]
    t = template.astype(np.float32)
    t = (t - t.mean()) * _hann2d(k)

    padded = np.zeros((h, w), dtype=np.float32)
    padded[:k, :k] = t

    s = search.astype(np.float32)
    s = s - s.mean()

    f1 = np.fft.rfft2(s)
    f2 = np.fft.rfft2(padded)
    r = f1 * np.conj(f2)
    r /= np.abs(r) + 1e-9
    return np.fft.irfft2(r, s=s.shape).astype(np.float32)


def phase_correlation_match(reference: np.ndarray, search: np.ndarray,
                            sizes=PHASE_SIZES,
                            method_name: str = "phase_correlation") -> MatchResult:
    """Best phase-correlation peak over a small sweep of template sizes.

    `score` is the ZNCC re-scored at the predicted location so it is directly
    comparable with every other matcher; the native phase-correlation peak
    sharpness (PSR) is kept in `extra`.
    """
    t0 = time.perf_counter()
    best = None
    for size in sizes:
        template = make_template(reference, int(size), 0.0, INNER_FRACTION)
        k = template.shape[0]
        if k >= min(search.shape):
            continue
        surface = phase_correlation_surface(search, template)
        idx = int(np.argmax(surface))
        py, px = np.unravel_index(idx, surface.shape)
        psr = peak_to_sidelobe_ratio(surface, int(px), int(py))
        dx, dy = subpixel_peak(surface, int(px), int(py))
        cx = float(px + dx + k / 2.0)
        cy = float(py + dy + k / 2.0)
        zncc = zncc_score_at(search, template, cx, cy)
        if best is None or psr > best["psr"]:
            best = {"x": cx, "y": cy, "psr": psr, "zncc": zncc,
                    "size": int(size), "peak": float(surface[py, px])}

    if best is None:
        raise ValueError("phase correlation: no usable template size")

    return MatchResult(
        x=best["x"], y=best["y"], score=best["zncc"], method=method_name,
        size=best["size"],
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"native_score": best["psr"], "native_metric": "psr",
               "raw_peak": best["peak"]},
    )

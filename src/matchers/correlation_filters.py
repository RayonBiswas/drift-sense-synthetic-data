"""MACE and OTDF correlation filters. The discriminator.

A plain matched filter maximizes signal-to-noise ratio, which is the wrong
objective when the problem is not noise but *self-similarity*: on a periodic
array the true location and its lattice neighbours all produce a strong
response, so a broad correlation peak is useless.

MACE (Minimum Average Correlation Energy) optimizes a different thing. It
minimizes the total energy of the correlation plane subject to hard
constraints that the response equals 1 at the origin for every training
view:

    h = D^-1 X (X^+ D^-1 X)^-1 u

with X the matrix of vectorized training spectra, D the diagonal of their
average power spectrum, and u the vector of desired peak values. Everything
that is not the peak is actively suppressed, so what comes out is a very
sharp, very selective spike -- which is exactly the tool for "several
candidates look alike, which one is the target?".

OTDF (optimal trade-off) is the same construction with D replaced by a blend

    T = (1 - alpha) * D + alpha * C

where C is the white-noise covariance. alpha = 0 gives pure MACE, maximally
sharp but brittle under noise; alpha -> 1 walks back toward the noise-tolerant
matched filter. One implementation, two registry entries, one honest knob.

The filter is trained on the reference alone -- rotated and rescaled views of
it -- so nothing about the search image or the ground truth leaks into it.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .common import (
    INNER_FRACTION,
    MatchResult,
    make_template,
    peak_to_sidelobe_ratio,
    subpixel_peak,
    top_k_peaks,
    zncc_score_at,
)

TRAIN_ANGLES = (-5.0, -2.5, 0.0, 2.5, 5.0)
TRAIN_SCALES = (0.9, 1.0, 1.1)
DEFAULT_BASE_SIZE = 100


def training_views(reference: np.ndarray, base_size: int = DEFAULT_BASE_SIZE,
                   angles=TRAIN_ANGLES, scales=TRAIN_SCALES) -> list:
    """Rotated/rescaled views of the reference, all resampled to one size.

    A filter trained on a single view is a matched filter for that view only.
    Training across the expected +-5 degree / +-15% envelope is what buys the
    tolerance -- at the cost of some peak sharpness, which is the classic
    MACE trade-off.
    """
    k = int(base_size * INNER_FRACTION)
    views = []
    for scale in scales:
        size = max(int(round(base_size * scale)), 16)
        for angle in angles:
            tpl = make_template(reference, size, angle, INNER_FRACTION)
            tpl = cv2.resize(tpl, (k, k), interpolation=cv2.INTER_AREA).astype(np.float32)
            tpl -= tpl.mean()
            std = tpl.std()
            if std > 1e-6:
                tpl /= std
            views.append(tpl)
    return views


def build_filter(views: list, alpha: float = 0.0) -> np.ndarray:
    """Synthesize the (spatial-domain, real) MACE/OTDF filter from views."""
    k = views[0].shape[0]
    d = k * k
    x = np.stack([np.fft.fft2(v).ravel() for v in views], axis=1)   # (d, n)

    power = (np.abs(x) ** 2).mean(axis=1)                            # diag(D)
    white = float(power.mean())
    t = (1.0 - alpha) * power + alpha * white
    t = np.maximum(t, 1e-8)

    y = x / t[:, None]                                               # T^-1 X
    gram = x.conj().T @ y                                            # X^+ T^-1 X
    gram += np.eye(gram.shape[0]) * (1e-6 * np.trace(gram).real / gram.shape[0])
    u = np.ones((gram.shape[0],), dtype=np.complex128)
    h_freq = (y @ np.linalg.solve(gram, u)).reshape(k, k)

    # Training views are real, so the solution keeps conjugate symmetry and
    # the spatial filter is real up to numerical noise.
    return np.real(np.fft.ifft2(h_freq)).astype(np.float32)


def correlation_plane(search: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """Correlate the filter over the whole search image in one FFT pass.

    The search image has its local mean removed first (box filter of the
    filter's own size). The filter is not brightness-invariant the way ZNCC
    is, and vignetting or a charging streak would otherwise dominate the
    response.
    """
    k = filt.shape[0]
    s = search.astype(np.float32)
    s = s - cv2.boxFilter(s, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)

    h, w = s.shape
    padded = np.zeros((h, w), dtype=np.float32)
    padded[:k, :k] = filt

    plane = np.fft.irfft2(np.fft.rfft2(s) * np.conj(np.fft.rfft2(padded)), s=(h, w))
    # Positions where the template would hang off the edge are meaningless.
    plane[h - k + 1:, :] = plane.min()
    plane[:, w - k + 1:] = plane.min()
    return plane.astype(np.float32)


def mace_match(reference: np.ndarray, search: np.ndarray,
               alpha: float = 0.0, base_size: int = DEFAULT_BASE_SIZE,
               top_k: int = 5, method_name: str = None) -> MatchResult:
    """Whole-image MACE/OTDF detection, ranked by peak-to-sidelobe ratio.

    `score` is the ZNCC re-scored at the winning location so it is comparable
    with the other matchers; the filter's own PSR is in `extra`.
    """
    t0 = time.perf_counter()
    if method_name is None:
        method_name = "mace" if alpha <= 1e-9 else f"otdf_a{alpha:g}"

    views = training_views(reference, base_size)
    filt = build_filter(views, alpha=alpha)
    plane = correlation_plane(search, filt)

    k = filt.shape[0]
    template = make_template(reference, base_size, 0.0, INNER_FRACTION)

    best = None
    for px, py, _ in top_k_peaks(plane, k=top_k, min_distance=max(k // 3, 4)):
        psr = peak_to_sidelobe_ratio(plane, px, py, exclude=max(k // 8, 3),
                                     window=max(k // 2, 12))
        dx, dy = subpixel_peak(plane, px, py)
        cx = float(px + dx + k / 2.0)
        cy = float(py + dy + k / 2.0)
        if best is None or psr > best["psr"]:
            best = {"x": cx, "y": cy, "psr": psr,
                    "zncc": zncc_score_at(search, template, cx, cy)}

    return MatchResult(
        x=best["x"], y=best["y"], score=best["zncc"], method=method_name,
        size=base_size,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"native_score": best["psr"], "native_metric": "psr",
               "alpha": alpha, "n_training_views": len(views)},
    )


def otdf_match(reference: np.ndarray, search: np.ndarray,
               alpha: float = 0.3, **kwargs) -> MatchResult:
    """Optimal-trade-off variant: some peak sharpness traded for noise tolerance."""
    return mace_match(reference, search, alpha=alpha, method_name=f"otdf_a{alpha:g}", **kwargs)


def rank_candidates(reference: np.ndarray, search: np.ndarray, candidates: list,
                    alpha: float = 0.0, base_size: int = DEFAULT_BASE_SIZE,
                    radius: int = 12) -> list:
    """Re-rank existing candidates by filter PSR. The pipeline's discriminator.

    Returns [{"x", "y", "psr", "plane_value"}, ...] sorted best first. Building
    the filter is the expensive part and it is built once for all candidates.
    """
    views = training_views(reference, base_size)
    filt = build_filter(views, alpha=alpha)
    plane = correlation_plane(search, filt)
    k = filt.shape[0]

    ranked = []
    for cx, cy, *_ in candidates:
        # the plane is indexed by template top-left, not centre
        tx = int(round(cx - k / 2.0))
        ty = int(round(cy - k / 2.0))
        x0 = int(np.clip(tx - radius, 0, plane.shape[1] - 1))
        y0 = int(np.clip(ty - radius, 0, plane.shape[0] - 1))
        x1 = int(np.clip(tx + radius + 1, 1, plane.shape[1]))
        y1 = int(np.clip(ty + radius + 1, 1, plane.shape[0]))
        window = plane[y0:y1, x0:x1]
        if window.size == 0:
            continue
        iy, ix = np.unravel_index(int(np.argmax(window)), window.shape)
        px, py = x0 + ix, y0 + iy
        ranked.append({
            "x": float(px + k / 2.0),
            "y": float(py + k / 2.0),
            "psr": peak_to_sidelobe_ratio(plane, px, py, exclude=max(k // 8, 3),
                                          window=max(k // 2, 12)),
            "plane_value": float(plane[py, px]),
        })
    ranked.sort(key=lambda r: r["psr"], reverse=True)
    return ranked

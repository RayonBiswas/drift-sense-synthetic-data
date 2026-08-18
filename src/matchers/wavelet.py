"""Wavelet-domain coarse-to-fine localization. The scout.

Idea: a brute-force ZNCC sweep evaluates every position at every scale and
angle at full resolution. Most of that work is spent proving that obviously
wrong places are wrong. A Haar pyramid lets us disqualify them 16x cheaper
per level, and the low-pass (LL) subband is exactly the representation that
suppresses two of our worst problems at once:

  * high-frequency detector noise averages away in the LL subband;
  * the finest periodic texture -- the repeat that makes a DRAM array
    confusable with itself -- is a *high*-frequency structure, so coarse
    scales are dominated by the larger-scale layout instead.

What it does not do is solve rotation or scale mismatch; that is
Fourier-Mellin's job. This stage's honest output is "the target is probably
in one of these few places", which is why it also returns candidates.

The transform is a plain orthonormal Haar DWT written against numpy, so it
adds no dependency: LL = (a+b+c+d)/2 over each 2x2 block, and the three
detail subbands are the corresponding sum/difference combinations.
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
    count_candidates,
    make_template,
    top_k_peaks,
)
from .zncc import zncc_refine, zncc_surface

# Scale differences smaller than this are invisible once downsampled, so the
# coarse stage sweeps a deliberately thin set of sizes and only 0 degrees.
COARSE_SIZES = (80, 100, 120)
DEFAULT_LEVELS = 2
DEFAULT_TOP_K = 6


def haar_dwt2(img: np.ndarray) -> tuple:
    """One level of orthonormal 2-D Haar. Returns (LL, LH, HL, HH), half size."""
    f = img.astype(np.float32)
    h, w = f.shape
    f = f[: h - (h % 2), : w - (w % 2)]
    a = f[0::2, 0::2]
    b = f[0::2, 1::2]
    c = f[1::2, 0::2]
    d = f[1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh


def haar_pyramid(img: np.ndarray, levels: int) -> list:
    """[level0, level1, ...] of LL subbands; level0 is the input as float32."""
    out = [img.astype(np.float32)]
    cur = out[0]
    for _ in range(int(levels)):
        cur = haar_dwt2(cur)[0]
        out.append(cur)
    return out


def detail_energy(img: np.ndarray, levels: int = 2) -> list:
    """Mean |detail| per level -- how much of the image is fine structure.

    Useful as a diagnostic: a heavily noised search image has abnormally high
    level-0 detail energy, which is precisely the energy the coarse stage
    throws away.
    """
    energies = []
    cur = img.astype(np.float32)
    for _ in range(int(levels)):
        ll, lh, hl, hh = haar_dwt2(cur)
        energies.append(float((np.abs(lh) + np.abs(hl) + np.abs(hh)).mean() / 3.0))
        cur = ll
    return energies


def wavelet_candidates(reference: np.ndarray, search: np.ndarray,
                       levels: int = DEFAULT_LEVELS,
                       top_k: int = DEFAULT_TOP_K,
                       coarse_sizes=COARSE_SIZES,
                       angle: float = 0.0,
                       peak_spacing_fraction: float = 0.25) -> dict:
    """Coarse localization only. Returns candidate centres in FULL-RES pixels.

    Each candidate is (x, y, coarse_score). Also returns `n_candidates`, the
    number of coarse peaks within 5% of the best -- the cheap, honest measure
    of how self-similar this particular search image is.

    `peak_spacing_fraction` sets the non-maximum-suppression radius as a
    fraction of the coarse template. It must stay *below* the array pitch or
    the suppression eats the lattice neighbours and the stage reports a
    confident single peak on an image that is in fact deeply ambiguous --
    which is the one failure mode this whole pipeline exists to catch.
    """
    t0 = time.perf_counter()
    factor = 2 ** int(levels)
    search_ll = haar_pyramid(search, levels)[-1]

    best = None
    best_surface = None
    for size in coarse_sizes:
        tpl_full = make_template(reference, int(size), angle, INNER_FRACTION)
        tpl_ll = haar_pyramid(tpl_full, levels)[-1]
        if tpl_ll.shape[0] < 6 or tpl_ll.shape[0] >= search_ll.shape[0]:
            continue
        surface = zncc_surface(search_ll, tpl_ll)
        _, score, _, _ = cv2.minMaxLoc(surface)
        if best is None or score > best["score"]:
            best = {"score": float(score), "size": int(size), "k": tpl_ll.shape[0]}
            best_surface = surface

    if best is None:
        raise ValueError("wavelet stage: no usable template size at this level")

    k = best["k"]
    spacing = max(int(k * peak_spacing_fraction), 2)
    peaks = top_k_peaks(best_surface, k=top_k, min_distance=spacing)
    candidates = [
        ((px + k / 2.0) * factor, (py + k / 2.0) * factor, score)
        for px, py, score in peaks
    ]
    return {
        "candidates": candidates,
        "best_score": best["score"],
        "best_size": best["size"],
        "peak_spacing_px": spacing * factor,
        "n_candidates": count_candidates(best_surface, best["score"],
                                         min_distance=spacing),
        "levels": int(levels),
        "surface": best_surface,
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
    }


def wavelet_match(reference: np.ndarray, search: np.ndarray,
                  levels: int = DEFAULT_LEVELS,
                  top_k: int = DEFAULT_TOP_K,
                  sizes=DEFAULT_SIZES, angles=DEFAULT_ANGLES,
                  method_name: str = "wavelet_zncc") -> MatchResult:
    """Wavelet scout + full-resolution ZNCC verification of each candidate.

    This is the "Wavelet + ZNCC" row of the ablation: cheap coarse search,
    expensive check only where it might pay off.
    """
    t0 = time.perf_counter()
    coarse = wavelet_candidates(reference, search, levels=levels, top_k=top_k)
    radius = 2 ** int(levels) * 2 + 4

    best = None
    for cx, cy, coarse_score in coarse["candidates"]:
        refined = zncc_refine(reference, search, cx, cy, search_radius=radius,
                              sizes=sizes, angles=angles)
        if best is None or refined.score > best.score:
            best = refined
            best.extra["coarse_score"] = coarse_score

    best.method = method_name
    best.n_candidates = coarse["n_candidates"]
    best.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    best.extra.update({
        "coarse_best_score": coarse["best_score"],
        "levels": coarse["levels"],
        "candidates_examined": len(coarse["candidates"]),
        "coarse_ms": round(coarse["elapsed_ms"], 1),
    })
    return best

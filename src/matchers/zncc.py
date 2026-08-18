"""ZNCC -- zero-mean normalized cross-correlation. The judge.

`cv2.matchTemplate(..., TM_CCOEFF_NORMED)` *is* ZNCC: for every window
position it subtracts the local mean from both template and window before
correlating, then divides by their standard deviations. That normalization
is why it survives the brightness and contrast difference between two
separate captures, which a plain SSD or raw correlation would not.

Strength: the most accurate final verification we have.
Weakness: cost grows with search area x sizes x angles, and on a periodic
array the surface has many near-equal peaks one lattice step apart.
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
    crop_roi,
    make_template,
    subpixel_peak,
)


def zncc_surface(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    return cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)


def zncc_match(reference: np.ndarray, search: np.ndarray,
               sizes=DEFAULT_SIZES, angles=DEFAULT_ANGLES,
               inner_fraction: float = INNER_FRACTION,
               subpixel: bool = True,
               method_name: str = "zncc") -> MatchResult:
    """Exhaustive multi-scale, multi-angle ZNCC over the whole search image."""
    t0 = time.perf_counter()
    best = None
    best_surface = None

    for size in sizes:
        size = int(size)
        if size < 16:
            continue
        template = make_template(reference, size, 0.0, inner_fraction)
        if template.shape[0] >= search.shape[0] or template.shape[1] >= search.shape[1]:
            continue
        for angle in angles:
            tpl = template if angle == 0.0 else make_template(reference, size, angle, inner_fraction)
            surface = zncc_surface(search, tpl)
            _, score, _, loc = cv2.minMaxLoc(surface)
            if best is None or score > best["score"]:
                best = {"score": float(score), "loc": loc, "size": size,
                        "angle": float(angle), "k": tpl.shape[0]}
                best_surface = surface

    if best is None:
        raise ValueError("reference is larger than the search image at every swept scale")

    k = best["k"]
    x, y = best["loc"]
    if subpixel:
        dx, dy = subpixel_peak(best_surface, x, y)
    else:
        dx = dy = 0.0

    return MatchResult(
        x=float(x + dx + k / 2.0),
        y=float(y + dy + k / 2.0),
        score=best["score"],
        method=method_name,
        angle=best["angle"],
        size=best["size"],
        n_candidates=count_candidates(best_surface, best["score"]),
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"subpixel_dx": dx, "subpixel_dy": dy},
    )


def zncc_refine(reference: np.ndarray, search: np.ndarray,
                cx: float, cy: float, search_radius: int = 24,
                sizes=DEFAULT_SIZES, angles=DEFAULT_ANGLES,
                inner_fraction: float = INNER_FRACTION,
                base_angle: float = 0.0) -> MatchResult:
    """Full-resolution ZNCC restricted to a window around a proposed centre.

    This is the last stage of every pipeline: once a cheap stage has said
    *roughly here*, ZNCC decides exactly where, at what scale and angle.
    """
    t0 = time.perf_counter()
    max_k = int(max(sizes) * inner_fraction)
    half = int(search_radius + max_k // 2 + 4)
    roi, ox, oy = crop_roi(search, cx, cy, half)

    best = None
    best_surface = None
    for size in sizes:
        size = int(size)
        for angle in angles:
            tpl = make_template(reference, size, base_angle + angle, inner_fraction)
            if tpl.shape[0] >= roi.shape[0] or tpl.shape[1] >= roi.shape[1]:
                continue
            surface = zncc_surface(roi, tpl)
            _, score, _, loc = cv2.minMaxLoc(surface)
            if best is None or score > best["score"]:
                best = {"score": float(score), "loc": loc, "size": size,
                        "angle": float(base_angle + angle), "k": tpl.shape[0]}
                best_surface = surface

    if best is None:
        # ROI too small for any template: fall back to the proposal itself.
        return MatchResult(x=float(cx), y=float(cy), score=-1.0, method="zncc_refine",
                           elapsed_ms=(time.perf_counter() - t0) * 1000.0)

    k = best["k"]
    x, y = best["loc"]
    dx, dy = subpixel_peak(best_surface, x, y)
    return MatchResult(
        x=float(ox + x + dx + k / 2.0),
        y=float(oy + y + dy + k / 2.0),
        score=best["score"],
        method="zncc_refine",
        angle=best["angle"],
        size=best["size"],
        n_candidates=count_candidates(best_surface, best["score"]),
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"roi_origin": (ox, oy)},
    )

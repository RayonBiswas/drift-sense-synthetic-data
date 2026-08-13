"""
Reference-to-search localization.

`localize(reference, wide)` finds where a small high-magnification reference
image sits inside a 1000x1000 low-magnification search image, and returns the
centre in search-image pixels.

Method
------
ZNCC -- zero-mean normalized cross-correlation, which is what
`cv2.matchTemplate(..., TM_CCOEFF_NORMED)` computes: for every window position
it subtracts the local mean from both template and window before correlating,
then divides by their standard deviations. That normalization is why it survives
the brightness and contrast differences between two separate captures, which a
plain SSD or raw correlation would not.

The matcher is given no privileged information. It does not know the true scale,
rotation or position, so it sweeps a range of template sizes and rotations and
keeps the highest-correlating candidate.

Ambiguity
---------
A DRAM array is periodic, so the correlation surface has many near-equal peaks
one lattice step apart -- the *right pattern, the wrong repeat*. `n_candidates`
counts the distinct peaks within 5% of the best score, which is the honest
measure of how confusable a given case is. A high count next to a large error is
the signature of periodic-array confusion rather than of a matcher that simply
failed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# The reference footprint is 100 * scale search pixels with scale in
# [0.85, 1.15]; barrel distortion shrinks it a little further.
DEFAULT_SIZES = tuple(range(80, 126, 5))
# The reference sits within +-5 degrees of the search image.
DEFAULT_ANGLES = (-5.0, -2.5, 0.0, 2.5, 5.0)
# Rotation leaves undefined corners; keep the inscribed centre of the template.
INNER_FRACTION = 0.78
# Peaks this close to the best score count as rival candidates.
CANDIDATE_RATIO = 0.95
# Minimum separation between distinct peaks, in search pixels.
CANDIDATE_MIN_DISTANCE = 8


@dataclass
class LocalizationResult:
    center_x: float
    center_y: float
    score: float
    n_candidates: int
    size: int
    angle: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


def _as_gray(img: np.ndarray) -> np.ndarray:
    """Accept grayscale or 3-channel input; matching runs on intensity."""
    if img is None:
        raise ValueError("image is None")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _count_candidates(surface: np.ndarray, best: float) -> int:
    """Distinct correlation peaks within CANDIDATE_RATIO of the best score.

    Non-maximum suppression by grayscale dilation: a pixel is a local peak if it
    equals the maximum of its neighbourhood. Counting these says how many places
    in the search image look about as much like the reference as the winner
    does -- on a periodic array, usually several.
    """
    if not np.isfinite(best):
        return 0
    threshold = best * CANDIDATE_RATIO if best > 0 else best
    k = 2 * CANDIDATE_MIN_DISTANCE + 1
    dilated = cv2.dilate(surface, np.ones((k, k), np.uint8))
    peaks = (surface >= dilated - 1e-6) & (surface >= threshold)
    return int(np.count_nonzero(peaks))


def localize(reference: np.ndarray, wide: np.ndarray,
             sizes=DEFAULT_SIZES, angles=DEFAULT_ANGLES) -> LocalizationResult:
    """Locate `reference` inside `wide`.

    Returns the best candidate's centre in `wide` pixel coordinates, its ZNCC
    score in [-1, 1], and how many rival peaks were within 5% of it.
    """
    ref = _as_gray(reference)
    search = _as_gray(wide)
    sh, sw = search.shape

    best = None
    best_surface = None

    for size in sizes:
        size = int(size)
        if size < 16:
            continue
        scaled = cv2.resize(ref, (size, size), interpolation=cv2.INTER_AREA)
        k = max(int(size * INNER_FRACTION), 8)
        if k >= sh or k >= sw:
            continue
        off = (size - k) // 2

        for angle in angles:
            if angle:
                m = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), float(angle), 1.0)
                rotated = cv2.warpAffine(scaled, m, (size, size), flags=cv2.INTER_LINEAR)
            else:
                rotated = scaled
            template = rotated[off:off + k, off:off + k]

            surface = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(surface)
            if best is None or score > best["score"]:
                best = {"score": float(score), "loc": loc, "size": size,
                        "angle": float(angle), "k": k}
                best_surface = surface

    if best is None:
        raise ValueError("reference is larger than the search image at every "
                         "swept scale; nothing could be matched")

    k = best["k"]
    cx = best["loc"][0] + k / 2.0
    cy = best["loc"][1] + k / 2.0
    half = best["size"] / 2.0

    return LocalizationResult(
        center_x=float(cx),
        center_y=float(cy),
        score=best["score"],
        n_candidates=_count_candidates(best_surface, best["score"]),
        size=best["size"],
        angle=best["angle"],
        x1=float(cx - half), y1=float(cy - half),
        x2=float(cx + half), y2=float(cy + half),
    )

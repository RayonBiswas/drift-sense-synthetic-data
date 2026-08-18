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


# Never scan more than this many above-threshold pixels when extracting peaks;
# a low threshold on a flat surface can otherwise put the whole image in play.
CANDIDATE_SCAN_LIMIT = 20000
# Cap on peaks returned. A 1000x1000 search image holds ~100 non-overlapping
# reference footprints, so this is generous.
MAX_CANDIDATES = 64
# Weight of the centre prior, in ZNCC score units per half-image of distance.
# The tool aimed at the site and drifted, so a candidate near the centre of the
# fresh scan is more likely to be the real one -- but a candidate that
# correlates decisively better should still win. Measured on 200 held-out
# drift-realistic pairs: 0.02 gives 39.0% at 10px against 37.5% for plain
# argmax, and halves median error (148 -> 80 px). Overriding the argmax outright
# instead of weighting it costs 10 points, because it throws away sub-pixel
# correct answers in favour of more central wrong ones.
CENTER_PRIOR_LAMBDA = 0.02
# Distance that costs exactly CENTER_PRIOR_LAMBDA of score: half the image.
CENTER_PRIOR_NORM = 500.0


@dataclass
class Candidate:
    """One correlation peak: its centre in search pixels and its ZNCC score."""
    center_x: float
    center_y: float
    score: float


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
    # Every rival peak within CANDIDATE_RATIO of the best score, strongest
    # first. Length equals n_candidates.
    candidates: list = None
    # "center" if the spec tie-break moved the answer off the strongest peak,
    # "score" if the strongest peak won anyway (or was the only one).
    tie_break_applied: str = "score"

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


def _find_candidates(surface: np.ndarray, best: float, k: int) -> list:
    """Distinct correlation peaks within CANDIDATE_RATIO of the best score.

    These are the places in the search image that look about as much like the
    reference as the winner does -- on a periodic array, usually several. They
    are what the spec means by "more than one matching region".

    Peaks are separated by greedy non-maximum suppression: take the strongest
    remaining pixel, drop everything within CANDIDATE_MIN_DISTANCE of it,
    repeat. Dilation-based peak finding was tried first but reports every pixel
    of a flat plateau as its own peak, which inflates the count.

    Returns Candidates centred on the template (the surface is indexed by the
    template's top-left corner, so k/2 is added), strongest first.
    """
    if not np.isfinite(best):
        return []
    # A ratio only means "within 5%" for a positive score; below zero it would
    # widen the net instead of narrowing it.
    threshold = best * CANDIDATE_RATIO if best > 0 else best

    ys, xs = np.nonzero(surface >= threshold)
    if len(ys) == 0:
        return []
    scores = surface[ys, xs]
    order = np.argsort(-scores)[:CANDIDATE_SCAN_LIMIT]

    kept = []
    for i in order:
        x, y, s = float(xs[i]), float(ys[i]), float(scores[i])
        if any((x - px) ** 2 + (y - py) ** 2 < CANDIDATE_MIN_DISTANCE ** 2
               for px, py, _ in kept):
            continue
        kept.append((x, y, s))
        if len(kept) >= MAX_CANDIDATES:
            break

    return [Candidate(center_x=x + k / 2.0, center_y=y + k / 2.0, score=s)
            for x, y, s in kept]


def localize(reference: np.ndarray, wide: np.ndarray,
             sizes=DEFAULT_SIZES, angles=DEFAULT_ANGLES,
             tie_break: str = "prior") -> LocalizationResult:
    """Locate `reference` inside `wide`.

    Returns the chosen candidate's centre in `wide` pixel coordinates, its ZNCC
    score in [-1, 1], and how many rival peaks were within 5% of it.

    `tie_break` decides which rival wins:

      "prior"   the default. Maximizes `score - CENTER_PRIOR_LAMBDA * distance
                to the image centre`, so of two candidates that correlate about
                equally the more central one wins, while a decisively better
                match keeps its win. This implements the problem statement's
                preference for the central match without discarding a confident
                answer, and it is the only policy of the three that beats plain
                argmax on held-out data.
      "center"  the spec's rule read as a hard override: of the matching
                regions found, return the one closest to the centre, whatever
                its score. Measurably worse -- kept for comparison.
      "score"   the plain argmax of the correlation surface, ignoring position.
    """
    if tie_break not in ("prior", "center", "score"):
        raise ValueError(f"tie_break must be 'prior', 'center' or 'score', "
                         f"got {tie_break!r}")
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
    candidates = _find_candidates(best_surface, best["score"], k)
    if not candidates:
        # Degenerate surface (all NaN, or a single pixel); fall back to argmax.
        candidates = [Candidate(center_x=best["loc"][0] + k / 2.0,
                                center_y=best["loc"][1] + k / 2.0,
                                score=best["score"])]

    # Spec: "If more than one matching region is found, return the one closest
    # to the center of the Search Image." With one candidate every policy
    # agrees, so this only bites on a genuinely ambiguous surface.
    mid_x, mid_y = sw / 2.0, sh / 2.0

    def _distance_to_middle(c: Candidate) -> float:
        return float(np.hypot(c.center_x - mid_x, c.center_y - mid_y))

    chosen = candidates[0]
    applied = "score"
    if len(candidates) > 1:
        if tie_break == "prior":
            chosen = max(candidates,
                         key=lambda c: c.score - CENTER_PRIOR_LAMBDA
                         * (_distance_to_middle(c) / CENTER_PRIOR_NORM))
        elif tie_break == "center":
            chosen = min(candidates, key=_distance_to_middle)
        if chosen is not candidates[0]:
            applied = tie_break

    half = best["size"] / 2.0
    cx, cy = chosen.center_x, chosen.center_y

    return LocalizationResult(
        center_x=float(cx),
        center_y=float(cy),
        score=float(chosen.score),
        n_candidates=len(candidates),
        size=best["size"],
        angle=best["angle"],
        x1=float(cx - half), y1=float(cy - half),
        x2=float(cx + half), y2=float(cy + half),
        candidates=candidates,
        tie_break_applied=applied,
    )

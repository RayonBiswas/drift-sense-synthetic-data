"""Shared plumbing for every matcher in this package.

Every matcher takes the *same* two images -- a high-magnification reference
(1000x1000 @ 1 nm/px) and a low-magnification search image (1000x1000 @
10 nm/px) -- and returns the same `MatchResult`, so they can be swapped,
compared and chained without the caller knowing which one it is talking to.

The reference's true footprint in the search image is 100x100 px (the pixel
size ratio is 10x), which is where DEFAULT_SIZES comes from. No matcher is
told the true scale, rotation or position: they sweep.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np

# Reference footprint is 100 * scale search px, scale in [0.85, 1.15];
# barrel distortion shrinks it a little further.
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
class MatchResult:
    """One matcher's answer, plus everything needed to judge it later."""

    x: float
    y: float
    score: float
    method: str
    angle: float = 0.0
    size: int = 0                      # template side in search px (=> implied scale)
    n_candidates: int = 1              # rival peaks within CANDIDATE_RATIO of the best
    elapsed_ms: float = 0.0
    stages: list = field(default_factory=list)   # pipeline trace, one dict per stage
    extra: dict = field(default_factory=dict)    # method-specific diagnostics (psr, ...)

    @property
    def bbox(self) -> tuple:
        half = self.size / 2.0
        return (self.x - half, self.y - half, self.x + half, self.y + half)

    def error_against(self, gt_x: float, gt_y: float) -> dict:
        """Predicted-minus-actual shift, in search-image pixels."""
        dx = self.x - gt_x
        dy = self.y - gt_y
        return {"dx": dx, "dy": dy, "distance_px": float(np.hypot(dx, dy))}

    def as_row(self, gt_x=None, gt_y=None, tolerance_px: float = 5.0) -> dict:
        """Flat dict for tables/CSV. Includes hit/miss when ground truth is known."""
        row = {
            "method": self.method,
            "pred_x": round(self.x, 2),
            "pred_y": round(self.y, 2),
            "score": round(self.score, 4),
            "angle": self.angle,
            "size": self.size,
            "n_candidates": self.n_candidates,
            "time_ms": round(self.elapsed_ms, 1),
        }
        if gt_x is not None and gt_y is not None:
            err = self.error_against(gt_x, gt_y)
            row["dx"] = round(err["dx"], 2)
            row["dy"] = round(err["dy"], 2)
            row["error_px"] = round(err["distance_px"], 2)
            row["hit"] = bool(err["distance_px"] <= tolerance_px)
        return row


@contextmanager
def timed(result_holder: dict, key: str = "elapsed_ms"):
    """Wall-clock a block and write the milliseconds into `result_holder[key]`."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        result_holder[key] = (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def as_gray(img: np.ndarray) -> np.ndarray:
    """Accept grayscale or 3-channel input; matching runs on intensity."""
    if img is None:
        raise ValueError("image is None")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def preprocess(img: np.ndarray, mode: str = "none") -> np.ndarray:
    """Normalization applied identically to reference and search.

    "none"    -- raw intensity (ZNCC already removes mean/contrast itself)
    "clahe"   -- local contrast equalization, helps when vignetting or
                 charging makes brightness vary across the frame
    "zscore"  -- global zero-mean unit-variance, rescaled back to uint8
    "bandpass"-- difference-of-Gaussians; kills both the slow illumination
                 gradient and the finest noise grain
    """
    g = as_gray(img)
    if mode == "none":
        return g
    if mode == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    if mode == "zscore":
        f = g.astype(np.float32)
        f = (f - f.mean()) / (f.std() + 1e-6)
        return np.clip(f * 48.0 + 128.0, 0, 255).astype(np.uint8)
    if mode == "bandpass":
        f = g.astype(np.float32)
        dog = cv2.GaussianBlur(f, (0, 0), 1.0) - cv2.GaussianBlur(f, (0, 0), 6.0)
        dog = (dog - dog.mean()) / (dog.std() + 1e-6)
        return np.clip(dog * 48.0 + 128.0, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown preprocess mode: {mode}")


def make_template(ref: np.ndarray, size: int, angle: float = 0.0,
                  inner_fraction: float = INNER_FRACTION) -> np.ndarray:
    """Reference resized to `size` px, rotated by `angle`, centre-cropped.

    Cropping to the inscribed centre is what makes rotated templates legal:
    a rotated square leaves undefined corners that would otherwise correlate
    against nothing.
    """
    scaled = cv2.resize(ref, (size, size), interpolation=cv2.INTER_AREA)
    if angle:
        m = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), float(angle), 1.0)
        scaled = cv2.warpAffine(scaled, m, (size, size), flags=cv2.INTER_LINEAR)
    k = max(int(size * inner_fraction), 8)
    off = (size - k) // 2
    return scaled[off:off + k, off:off + k]


# --------------------------------------------------------------------------
# correlation-surface analysis
# --------------------------------------------------------------------------

def subpixel_peak(surface: np.ndarray, x: int, y: int) -> tuple:
    """Parabola fit through the 3 samples either side of the peak.

    The correlation surface is sampled on an integer grid but the true peak
    is not on it. Fitting a parabola in x and y independently recovers the
    fractional part -- typically worth a few tenths of a pixel.
    """
    h, w = surface.shape
    dx = dy = 0.0
    if 0 < x < w - 1:
        l, c, r = float(surface[y, x - 1]), float(surface[y, x]), float(surface[y, x + 1])
        denom = l - 2 * c + r
        if abs(denom) > 1e-9:
            dx = float(np.clip(0.5 * (l - r) / denom, -1.0, 1.0))
    if 0 < y < h - 1:
        u, c, d = float(surface[y - 1, x]), float(surface[y, x]), float(surface[y + 1, x])
        denom = u - 2 * c + d
        if abs(denom) > 1e-9:
            dy = float(np.clip(0.5 * (u - d) / denom, -1.0, 1.0))
    return dx, dy


def top_k_peaks(surface: np.ndarray, k: int = 8,
                min_distance: int = CANDIDATE_MIN_DISTANCE) -> list:
    """The k strongest well-separated local maxima, best first.

    Non-maximum suppression by grayscale dilation: a pixel is a local peak if
    it equals the maximum of its neighbourhood. On a periodic array this is
    how you see the rivals instead of only the winner.
    """
    ksz = 2 * int(min_distance) + 1
    dilated = cv2.dilate(surface, np.ones((ksz, ksz), np.uint8))
    peaks = surface >= dilated - 1e-6
    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        _, _, _, loc = cv2.minMaxLoc(surface)
        return [(int(loc[0]), int(loc[1]), float(surface[loc[1], loc[0]]))]
    scores = surface[ys, xs]
    order = np.argsort(scores)[::-1][:k]
    return [(int(xs[i]), int(ys[i]), float(scores[i])) for i in order]


def count_candidates(surface: np.ndarray, best: float,
                     ratio: float = CANDIDATE_RATIO,
                     min_distance: int = CANDIDATE_MIN_DISTANCE) -> int:
    """How many distinct peaks are within `ratio` of the best score.

    A high count next to a large error is the signature of periodic-array
    confusion -- the right pattern, the wrong repeat -- rather than of a
    matcher that simply failed.
    """
    if not np.isfinite(best):
        return 0
    threshold = best * ratio if best > 0 else best
    ksz = 2 * int(min_distance) + 1
    dilated = cv2.dilate(surface, np.ones((ksz, ksz), np.uint8))
    peaks = (surface >= dilated - 1e-6) & (surface >= threshold)
    return int(np.count_nonzero(peaks))


def peak_to_sidelobe_ratio(surface: np.ndarray, x: int, y: int,
                           exclude: int = 5, window: int = 25) -> float:
    """PSR = (peak - mean(sidelobe)) / std(sidelobe).

    The standard correlation-filter sharpness measure. Unlike a raw
    correlation value it says how *isolated* the peak is, which is exactly
    what a MACE filter is optimized to maximize and exactly what a periodic
    background destroys.
    """
    h, w = surface.shape
    y0, y1 = max(y - window, 0), min(y + window + 1, h)
    x0, x1 = max(x - window, 0), min(x + window + 1, w)
    region = surface[y0:y1, x0:x1].astype(np.float64).copy()
    peak = float(surface[y, x])
    ey0, ey1 = max(y - exclude - y0, 0), min(y + exclude + 1 - y0, region.shape[0])
    ex0, ex1 = max(x - exclude - x0, 0), min(x + exclude + 1 - x0, region.shape[1])
    mask = np.ones(region.shape, dtype=bool)
    mask[ey0:ey1, ex0:ex1] = False
    side = region[mask]
    if side.size < 8:
        return 0.0
    return float((peak - side.mean()) / (side.std() + 1e-9))


def zncc_score_at(search: np.ndarray, template: np.ndarray,
                  cx: float, cy: float) -> float:
    """ZNCC of `template` against the search patch centred on (cx, cy).

    Used to re-score a candidate that some other stage proposed, so that all
    stages end up on one comparable scale.
    """
    k = template.shape[0]
    x0 = int(round(cx - k / 2.0))
    y0 = int(round(cy - k / 2.0))
    x0 = int(np.clip(x0, 0, search.shape[1] - k))
    y0 = int(np.clip(y0, 0, search.shape[0] - k))
    patch = search[y0:y0 + k, x0:x0 + template.shape[1]]
    if patch.shape != template.shape:
        return -1.0
    a = patch.astype(np.float32)
    b = template.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt(float((a * a).sum()) * float((b * b).sum())) + 1e-9
    return float((a * b).sum() / denom)


def crop_roi(img: np.ndarray, cx: float, cy: float, half: int) -> tuple:
    """Clamped square crop around (cx, cy). Returns (patch, x0, y0)."""
    h, w = img.shape[:2]
    x0 = int(np.clip(round(cx) - half, 0, max(w - 1, 0)))
    y0 = int(np.clip(round(cy) - half, 0, max(h - 1, 0)))
    x1 = int(np.clip(round(cx) + half, 1, w))
    y1 = int(np.clip(round(cy) + half, 1, h))
    return img[y0:y1, x0:x1], x0, y0

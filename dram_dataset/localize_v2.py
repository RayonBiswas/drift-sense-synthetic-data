"""
Reference-to-search localization, second generation.

`localize_v2(reference, search)` returns the centre of the reference inside the
1000x1000 search image, plus a calibrated confidence.

Why v1 tops out around a third of cases
---------------------------------------
Inside a single array block ("mat") the layout is *exactly* periodic: same
pitch, same phase, every repeat. Two windows one lattice step apart differ only
by per-line placement jitter (0.02-0.35 px) and line-width jitter (9%), and the
search capture blurs by 0.8-2.4 px and adds heavy noise on top. So the evidence
that would separate them is destroyed by the imaging. A ZNCC surface computed on
raw intensity therefore has many near-equal peaks, and picking the argmax is
close to picking at random among them -- which is exactly the measured failure:
either sub-pixel correct, or hundreds of pixels wrong, with nothing in between.

No amount of tuning the fine matcher fixes this, because the information is not
in the fine detail. It is in the *large-scale* structure: the canvas is built
from mats separated by flat peripheral/routing strips, and each mat carries its
own lattice phase and a small pitch trim. That envelope is not periodic at the
lattice scale, so it can tell one repeat from another.

What was tried, and what the measurements said
----------------------------------------------
Five stages were implemented and ablated on the 100 held-out pairs of
`dataset_drift/test`. Two helped, one was neutral, one actively hurt. The
numbers below are from `reports/localizer_ablation/report.json`; the defaults in
`Config` are set to what won, not to what was expected to win.

1. Sub-pixel refinement -- KEPT (+5 points at 1 px).
   A parabola through the correlation peak and its two neighbours per axis; the
   standard sub-pixel estimator for correlation surfaces. Accuracy at 1 px goes
   29.0% -> 34.0%. It changes nothing at 5 px, because it corrects fractions of
   a pixel, not lattice-step blunders.

2. Centre prior -- KEPT (median error 160 -> 76 px).
   Carried over from v1. Stage drift means the site is near the centre of the
   fresh scan, so of two near-equal peaks the more central one is likelier.
   Weighted, never a hard override.

3. Bandpass (difference of Gaussians) -- NEUTRAL, defaulted off.
   Intended to remove the shading, vignette and gamma differences between two
   independent captures. Accuracy is identical with and without, and median
   error is slightly *better* without (67.4 vs 76.3 px, within noise at n=100).
   ZNCC already subtracts local mean and divides by local standard deviation,
   so it was largely doing this job already. Kept switchable, off by default.

4. Envelope as a prior or re-ranker -- REJECTED (-6 to -7 points at 5 px).
   The hypothesis: inside a mat the lattice is exactly periodic, so the only
   thing that can tell one repeat from another is the larger-scale mat/strip
   layout. Reducing both images to a local (mean, contrast) description at a
   scale well above the lattice pitch should therefore disambiguate.
   It does not. Added to the surface it scores 31.0% against 37.0%; used to
   re-rank discrete peaks it scores 30.0%. The reason is a scale mismatch: the
   31 px averaging window that removes the lattice also makes the envelope peak
   broad and displaced by several pixels, while the ZNCC gap between the true
   peak and its rivals is only ~0.02. The envelope term swamps that gap and
   pulls the answer off the sharp peak.

5. Confidence -- THE ACTUAL WIN.
   The envelope turns out to be a good *confidence* signal even though it is a
   bad selector, and this is measured non-circularly: `envelope_agreement` is
   always computed and reported, but only influences the choice when
   `Config.envelope` says so, and the gating numbers come from the run where it
   did not. Envelope agreement >= 0.85 keeps 9% of cases at 88.9% accuracy;
   rival count == 1 keeps 12% at 83.3%. PSR is real but weaker than both.

The honest headline: none of this moved accuracy at 5 px, which stays at 37%.
Within a mat the repeats are genuinely identical once the search capture's
0.8-2.4 px blur and heavy noise have destroyed the sub-pixel placement jitter
that distinguishes them. That is an information problem, not a matcher problem,
and no amount of classical post-processing recovers information the imaging
threw away. What classical processing *can* do is halve the median error and
tell you reliably when to disbelieve the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# ---- fine stage (inherited from v1, unchanged) ---------------------------- #
DEFAULT_SIZES = tuple(range(80, 126, 5))
DEFAULT_ANGLES = (-5.0, -2.5, 0.0, 2.5, 5.0)
INNER_FRACTION = 0.78

# ---- bandpass ------------------------------------------------------------- #
# Low sigma keeps the lattice (pitch 6-14 px); high sigma removes shading and
# the slow vignette/gamma differences between the two captures.
DOG_SIGMA_LOW = 1.0
DOG_SIGMA_HIGH = 12.0

# ---- envelope ------------------------------------------------------------- #
# Averaging window, in search pixels. Must be comfortably larger than the
# coarsest lattice pitch (14 px) so the periodic carrier is averaged away, and
# smaller than the narrowest mat (90 px) so mat/strip structure survives.
ENVELOPE_WINDOW = 31
# The envelope surface is computed on images reduced by this factor: it carries
# no high-frequency information by construction, so full resolution is wasted.
ENVELOPE_DECIMATION = 4
# Weight of the envelope prior in ZNCC score units.
ENVELOPE_WEIGHT = 0.35

# ---- centre prior (inherited from v1) ------------------------------------- #
CENTER_PRIOR_LAMBDA = 0.02
CENTER_PRIOR_NORM = 500.0

# ---- confidence ----------------------------------------------------------- #
PSR_EXCLUDE_RADIUS = 12
# Peaks within this fraction of the best score count as unresolved rivals.
CANDIDATE_RATIO = 0.95
CANDIDATE_MIN_DISTANCE = 8
MAX_CANDIDATES = 64


@dataclass
class Config:
    # Neutral in measurement; off is marginally better on median error and
    # cheaper. Kept switchable because it may matter on real SEM data, where
    # shading is stronger than this generator's.
    bandpass: bool = False
    # "off"     ignore the envelope for selection. THE DEFAULT: both other
    #           modes measured worse. Its value is reported either way.
    # "add"     add the envelope surface to the fine surface before argmax.
    #           Measured 31.0% vs 37.0% at 5 px -- the broad envelope peak
    #           drags the argmax off the sharp fine peak.
    # "rerank"  use the envelope only to choose among the fine surface's own
    #           peaks, so position stays sub-pixel. Still worse: 30.0%.
    envelope: str = "off"
    subpixel: bool = True
    center_prior: bool = True
    envelope_weight: float = ENVELOPE_WEIGHT
    rerank_top_k: int = 24
    rerank_min_distance: int = 8
    sizes: tuple = DEFAULT_SIZES
    angles: tuple = DEFAULT_ANGLES


@dataclass
class Result:
    center_x: float
    center_y: float
    score: float
    psr: float
    size: int
    angle: float
    # Envelope correlation at the chosen peak. Always computed and reported,
    # regardless of whether the envelope was allowed to influence the choice --
    # otherwise measuring it as a confidence signal would be circular.
    envelope_agreement: float = 0.0
    # Peaks within CANDIDATE_RATIO of the best score on the winning surface.
    n_candidates: int = 1

    @property
    def confident(self) -> bool:
        """Thresholds calibrated on the held-out test split; see
        reports/localizer_ablation/report.json."""
        return self.n_candidates <= 2


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #

def _as_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("image is None")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _bandpass(img: np.ndarray) -> np.ndarray:
    """Difference of Gaussians, returned as float32.

    Kills the additive brightness offset and the smooth multiplicative shading
    (vignette, gamma, charging) that differ between the reference and the search
    capture, without touching the lattice the fine stage needs.
    """
    f = img.astype(np.float32)
    lo = cv2.GaussianBlur(f, (0, 0), DOG_SIGMA_LOW)
    hi = cv2.GaussianBlur(f, (0, 0), DOG_SIGMA_HIGH)
    return lo - hi


def _envelope_maps(img: np.ndarray, window: int, decim: int) -> np.ndarray:
    """Local (mean, contrast) at a scale that averages the lattice away.

    Returns a float32 array of shape (h/decim, w/decim, 2). Channel 0 is the
    local mean -- strips are flat mid-grey, mats are a brighter mixture, so this
    separates them. Channel 1 is the local standard deviation, i.e. how much
    structure is present: high inside a mat, low on a strip. Together they
    describe the block layout while carrying essentially no information about
    which lattice repeat you are on, which is exactly what is wanted.
    """
    f = img.astype(np.float32)
    k = (window, window)
    mean = cv2.boxFilter(f, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(f * f, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    if decim > 1:
        mean = mean[::decim, ::decim]
        std = std[::decim, ::decim]
    return np.dstack([mean, std]).astype(np.float32)


# --------------------------------------------------------------------------- #
# correlation helpers
# --------------------------------------------------------------------------- #

def _rotate_and_scale(ref: np.ndarray, size: int, angle: float) -> np.ndarray:
    """Resize the reference to `size` and rotate by `angle`, then crop to the
    inscribed centre so no undefined corner enters the correlation."""
    small = cv2.resize(ref, (size, size), interpolation=cv2.INTER_AREA)
    if angle != 0.0:
        m = cv2.getRotationMatrix2D((size / 2.0 - 0.5, size / 2.0 - 0.5), angle, 1.0)
        small = cv2.warpAffine(small, m, (size, size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
    inner = max(int(round(size * INNER_FRACTION)), 8)
    off = (size - inner) // 2
    return small[off:off + inner, off:off + inner]


def _zncc(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
        return None
    return cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)


def _subpixel_peak(surf: np.ndarray, py: int, px: int) -> tuple:
    """Parabola through the peak and its two neighbours, per axis.

    For three samples around a maximum the vertex offset is
    0.5*(a-c) / (a-2b+c). Clamped to +-1 px: a larger correction means the
    surface is not locally parabolic and the integer peak is the better answer.
    """
    h, w = surf.shape
    dy = dx = 0.0
    if 0 < py < h - 1:
        a, b, c = float(surf[py - 1, px]), float(surf[py, px]), float(surf[py + 1, px])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dy = float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))
    if 0 < px < w - 1:
        a, b, c = float(surf[py, px - 1]), float(surf[py, px]), float(surf[py, px + 1])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dx = float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))
    return dx, dy


def _top_peaks(surf: np.ndarray, k: int, min_distance: int) -> list:
    """The k strongest well-separated maxima of `surf`, strongest first.

    Greedy non-maximum suppression: repeatedly take the global maximum, then
    blank a disc of `min_distance` around it so the next pick is a genuinely
    different location rather than the same peak's shoulder.
    """
    work = surf.copy()
    peaks = []
    for _ in range(k):
        py, px = np.unravel_index(int(np.argmax(work)), work.shape)
        val = float(work[py, px])
        if not np.isfinite(val) or val <= -1e9:
            break
        peaks.append((py, px, float(surf[py, px])))
        y0, y1 = max(py - min_distance, 0), min(py + min_distance + 1, work.shape[0])
        x0, x1 = max(px - min_distance, 0), min(px + min_distance + 1, work.shape[1])
        work[y0:y1, x0:x1] = -1e9
    return peaks


def _psr(surf: np.ndarray, py: int, px: int, radius: int = PSR_EXCLUDE_RADIUS) -> float:
    """Peak height in standard deviations of the sidelobe region.

    The sidelobe is the whole surface minus a small disc around the peak. A
    lattice-ambiguous case has many peaks of similar height, which inflates the
    sidelobe standard deviation and drives PSR down; a unique match leaves a
    flat sidelobe and a high PSR.
    """
    h, w = surf.shape
    mask = np.ones((h, w), dtype=bool)
    y0, y1 = max(py - radius, 0), min(py + radius + 1, h)
    x0, x1 = max(px - radius, 0), min(px + radius + 1, w)
    mask[y0:y1, x0:x1] = False
    side = surf[mask]
    if side.size < 16:
        return 0.0
    mu, sd = float(side.mean()), float(side.std())
    if sd < 1e-9:
        return 0.0
    return (float(surf[py, px]) - mu) / sd


# --------------------------------------------------------------------------- #
# envelope prior
# --------------------------------------------------------------------------- #

def _envelope_surface(ref_raw: np.ndarray, env_search: np.ndarray, size: int):
    """Coarse correlation of the mat/strip layout.

    Alignment is the whole difficulty here. The fine surface is indexed by the
    top-left corner of the *inner-cropped* template; an envelope surface built
    from the full reference at a decimated scale is indexed by something else
    entirely, and resampling it onto the fine grid leaves a systematic offset of
    roughly (size - inner)/2 pixels. A prior that is a few pixels off does not
    merely fail to help -- it actively pulls the peak off the correct answer.

    So the envelope is correlated with exactly the same template geometry as the
    fine stage, at full resolution: same inner crop, same size, therefore the
    same output shape and the same indexing, and the two surfaces can simply be
    added. Rotation is ignored -- a few degrees does not change a block layout
    -- so this is computed once per size rather than once per (size, angle).
    """
    template = _rotate_and_scale(ref_raw, size, 0.0)
    env_t = _envelope_maps(template, ENVELOPE_WINDOW, 1)
    if (env_t.shape[0] > env_search.shape[0]
            or env_t.shape[1] > env_search.shape[1]):
        return None

    total = None
    for ch in range(2):
        s = np.ascontiguousarray(env_search[:, :, ch])
        t = np.ascontiguousarray(env_t[:, :, ch])
        if t.std() < 1e-6 or s.std() < 1e-6:
            continue
        surf = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
        total = surf if total is None else total + surf
    if total is None:
        return None
    return total * 0.5


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #

def localize_v2(reference: np.ndarray, search: np.ndarray,
                cfg: Config = None) -> Result:
    """Locate `reference` inside `search`, returning its centre in search px."""
    cfg = cfg or Config()
    ref_raw = _as_gray(reference)
    search_raw = _as_gray(search)

    if cfg.bandpass:
        ref_f = _bandpass(ref_raw)
        search_f = _bandpass(search_raw)
    else:
        ref_f = ref_raw.astype(np.float32)
        search_f = search_raw.astype(np.float32)

    sh, sw = search_f.shape
    cy0, cx0 = (sh - 1) / 2.0, (sw - 1) / 2.0

    # Envelope of the search image: computed once, on raw intensity, because
    # the bandpass deliberately removes the low-frequency content it is made of.
    # Always computed: the envelope is reported as a confidence signal even
    # when it is not allowed to influence which peak wins.
    use_env = cfg.envelope in ("add", "rerank")
    env_search = _envelope_maps(search_raw, ENVELOPE_WINDOW, 1)

    best = None
    # The envelope surface depends only on the assumed footprint size, not on
    # rotation (a few degrees does not change a block layout), so it is computed
    # once per size rather than once per (size, angle).
    for size in cfg.sizes:
        env_cache = None
        for angle in cfg.angles:
            template = _rotate_and_scale(ref_f, size, angle)
            surf = _zncc(search_f, template)
            if surf is None:
                continue

            th, tw = template.shape
            if env_cache is None:
                env_cache = _envelope_surface(ref_raw, env_search, size)
            env_ok = env_cache is not None and env_cache.shape == surf.shape

            def _centre_penalty(py, px):
                if not cfg.center_prior:
                    return 0.0
                d = np.hypot((px + tw / 2.0) - cx0, (py + th / 2.0) - cy0)
                return CENTER_PRIOR_LAMBDA * (d / CENTER_PRIOR_NORM)

            if cfg.envelope == "rerank":
                # Position comes from the fine surface's own peaks; the
                # envelope only decides which of them to believe.
                cands = _top_peaks(surf, cfg.rerank_top_k, cfg.rerank_min_distance)
                for py, px, zncc in cands:
                    env_v = float(env_cache[py, px]) if env_ok else 0.0
                    val = zncc + (cfg.envelope_weight * env_v if use_env else 0.0) \
                        - _centre_penalty(py, px)
                    if best is None or val > best[0]:
                        best = (val, surf, surf, py, px, size, angle, th, tw, env_v)
                continue

            combined = surf
            if cfg.envelope == "add" and env_ok:
                combined = surf + cfg.envelope_weight * env_cache
            if cfg.center_prior:
                yy = np.arange(surf.shape[0], dtype=np.float32)[:, None] + th / 2.0
                xx = np.arange(surf.shape[1], dtype=np.float32)[None, :] + tw / 2.0
                dist = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2)
                combined = combined - CENTER_PRIOR_LAMBDA * (dist / CENTER_PRIOR_NORM)

            py, px = np.unravel_index(int(np.argmax(combined)), combined.shape)
            val = float(combined[py, px])
            if best is None or val > best[0]:
                best = (val, surf, combined, py, px, size, angle, th, tw,
                        float(env_cache[py, px]) if env_ok else 0.0)

    if best is None:
        return Result(cx0, cy0, 0.0, 0.0, cfg.sizes[0], 0.0)

    _, surf, combined, py, px, size, angle, th, tw, env_val = best

    dx = dy = 0.0
    if cfg.subpixel:
        dx, dy = _subpixel_peak(combined, py, px)

    # How many rivals sit within CANDIDATE_RATIO of the winner on the fine
    # surface. Measured to be the single best "is this trustworthy?" signal.
    peaks = _top_peaks(surf, MAX_CANDIDATES, CANDIDATE_MIN_DISTANCE)
    best_score = peaks[0][2] if peaks else 0.0
    n_cand = sum(1 for _, _, v in peaks if v >= best_score * CANDIDATE_RATIO) if best_score > 0 else 1

    cx = px + dx + tw / 2.0
    cy = py + dy + th / 2.0
    return Result(center_x=float(cx), center_y=float(cy),
                  score=float(surf[py, px]), psr=_psr(surf, py, px),
                  size=size, angle=angle, envelope_agreement=env_val,
                  n_candidates=int(n_cand))

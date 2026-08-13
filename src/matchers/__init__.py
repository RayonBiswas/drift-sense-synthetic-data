"""Matcher registry -- every localization algorithm behind one name.

    from src.matchers import run_matcher, compare, METHODS
    result = run_matcher("adaptive", reference_img, search_img)
    rows   = compare(["zncc", "wavelet_zncc", "adaptive"], ref, srch, gt_x, gt_y)

Score convention
----------------
`MatchResult.score` is always a ZNCC-comparable confidence in [-1, 1],
re-scored at the predicted location for the methods that do not natively
produce one. Each method's own metric (phase-correlation PSR, MACE PSR,
RANSAC inlier count, Fourier-Mellin response) is kept in
`extra["native_score"]` with its name in `extra["native_metric"]`. Mixing the
two on one axis would make the comparison table meaningless, which is why
they are kept apart.

None of these matchers is given privileged information. They do not know the
true scale, rotation or position -- they sweep. Ground truth is used only
afterwards, to measure the error.
"""

from __future__ import annotations

import traceback

import numpy as np

from .adaptive import ABLATIONS, PipelineConfig, adaptive_match, explain
from .common import (
    CANDIDATE_MIN_DISTANCE,
    CANDIDATE_RATIO,
    DEFAULT_ANGLES,
    DEFAULT_SIZES,
    INNER_FRACTION,
    MatchResult,
    as_gray,
    preprocess,
)
from .correlation_filters import mace_match, otdf_match
from .features import feature_match
from .fourier_mellin import estimate_rotation_scale, fourier_mellin_match
from .phase import phase_correlation_match
from .wavelet import detail_energy, wavelet_candidates, wavelet_match
from .zncc import zncc_match, zncc_refine

__all__ = [
    "METHODS", "METHOD_INFO", "PIPELINE_METHODS", "SINGLE_METHODS",
    "run_matcher", "compare", "explain", "MatchResult", "PipelineConfig",
    "adaptive_match", "zncc_match", "wavelet_match", "phase_correlation_match",
    "fourier_mellin_match", "mace_match", "otdf_match", "feature_match",
    "estimate_rotation_scale", "wavelet_candidates", "detail_energy",
]


def _ablation(key: str):
    def run(reference, search, **kwargs):
        cfg = ABLATIONS[key]
        return adaptive_match(reference, search, config=cfg)
    return run


#: name -> callable(reference, search, **kwargs) -> MatchResult
METHODS = {
    "zncc": zncc_match,
    "wavelet_zncc": wavelet_match,
    "phase_correlation": phase_correlation_match,
    "fourier_mellin": fourier_mellin_match,
    "mace": mace_match,
    "otdf": otdf_match,
    "sift": lambda ref, srch, **kw: feature_match(ref, srch, detector="sift", **kw),
    "orb": lambda ref, srch, **kw: feature_match(ref, srch, detector="orb", **kw),
    "wavelet+zncc": _ablation("wavelet+zncc"),
    "wavelet+fm+zncc": _ablation("wavelet+fm+zncc"),
    "wavelet+mace+zncc": _ablation("wavelet+mace+zncc"),
    "adaptive": _ablation("adaptive"),
    "adaptive_all_stages": _ablation("adaptive_all_stages"),
}

SINGLE_METHODS = ["zncc", "wavelet_zncc", "phase_correlation", "fourier_mellin",
                  "mace", "otdf", "sift", "orb"]
PIPELINE_METHODS = ["wavelet+zncc", "wavelet+fm+zncc", "wavelet+mace+zncc",
                    "adaptive", "adaptive_all_stages"]

#: what each method is for -- shown in the UI so the comparison is readable
METHOD_INFO = {
    "zncc": {
        "label": "ZNCC (baseline)",
        "role": "Judge",
        "strength": "Most accurate final verification; immune to brightness/contrast shifts.",
        "weakness": "Cost scales with search area x sizes x angles; confused by periodic repeats.",
    },
    "wavelet_zncc": {
        "label": "Wavelet coarse-to-fine + ZNCC",
        "role": "Scout + judge",
        "strength": "Noise and fine periodic texture average away in the coarse LL subband; far cheaper.",
        "weakness": "Does not fix rotation or scale mismatch on its own.",
    },
    "phase_correlation": {
        "label": "Phase correlation",
        "role": "Fast translation estimate",
        "strength": "Two FFTs; sharp delta response for pure displacement.",
        "weakness": "Translation only; spectral whitening amplifies noise; splits on periodic scenes.",
    },
    "fourier_mellin": {
        "label": "Fourier-Mellin",
        "role": "Geometry specialist",
        "strength": "Recovers rotation and scale by turning them into a shift in log-polar space.",
        "weakness": "Needs a candidate patch to compare against; 180-degree ambiguity; noise-sensitive.",
    },
    "mace": {
        "label": "MACE filter",
        "role": "Discriminator",
        "strength": "Minimizes correlation-plane energy: a very sharp, very selective peak in clutter.",
        "weakness": "Filter synthesis cost; brittle under heavy noise (that is what OTDF relaxes).",
    },
    "otdf": {
        "label": "OTDF filter (trade-off)",
        "role": "Discriminator, noise-tolerant",
        "strength": "MACE sharpness traded back toward the matched filter's noise tolerance.",
        "weakness": "Broader peak than MACE, so less decisive between lattice repeats.",
    },
    "sift": {
        "label": "SIFT features + RANSAC",
        "role": "Fallback",
        "strength": "Handles rotation, scale and partial visibility from surviving local structure.",
        "weakness": "A repetitive array gives near-identical descriptors; the ratio test discards them.",
    },
    "orb": {
        "label": "ORB features + RANSAC",
        "role": "Fallback (fast)",
        "strength": "Much cheaper than SIFT; binary descriptors.",
        "weakness": "Even weaker than SIFT on low-texture periodic patterns.",
    },
    "wavelet+zncc": {
        "label": "Pipeline: wavelet -> ZNCC",
        "role": "Ablation",
        "strength": "The cheap pipeline: coarse search, full-resolution verification only.",
        "weakness": "No geometry correction, no discrimination stage.",
    },
    "wavelet+fm+zncc": {
        "label": "Pipeline: wavelet -> Fourier-Mellin -> ZNCC",
        "role": "Ablation",
        "strength": "Adds rotation/scale correction before the final score.",
        "weakness": "Still no answer to lattice ambiguity.",
    },
    "wavelet+mace+zncc": {
        "label": "Pipeline: wavelet -> MACE -> ZNCC",
        "role": "Ablation",
        "strength": "Adds discrimination between look-alike candidates.",
        "weakness": "Still no answer to rotation/scale mismatch.",
    },
    "adaptive": {
        "label": "Adaptive hybrid (recommended)",
        "role": "Full pipeline",
        "strength": "Every stage has an entry condition, so expensive stages run only on evidence.",
        "weakness": "Behaviour depends on the gate thresholds -- they are exposed, tune them.",
    },
    "adaptive_all_stages": {
        "label": "Adaptive, gates forced open",
        "role": "Cost reference",
        "strength": "Runs every stage on every image -- the upper bound on both cost and coverage.",
        "weakness": "Wasteful by construction; it exists to show what the gating saves.",
    },
}


def run_matcher(name: str, reference: np.ndarray, search: np.ndarray, **kwargs) -> MatchResult:
    """Run one registered matcher by name."""
    if name not in METHODS:
        raise KeyError(f"unknown matcher '{name}'. Known: {sorted(METHODS)}")
    return METHODS[name](as_gray(reference), as_gray(search), **kwargs)


def compare(names, reference: np.ndarray, search: np.ndarray,
            gt_x: float = None, gt_y: float = None,
            tolerance_px: float = 5.0, raise_errors: bool = False) -> list:
    """Run several matchers on the same pair and return one row each.

    A matcher that throws is reported as a failed row rather than taking the
    whole comparison down with it -- some of these methods legitimately fail
    on some inputs, and that is data.
    """
    rows = []
    for name in names:
        try:
            result = run_matcher(name, reference, search)
            row = result.as_row(gt_x, gt_y, tolerance_px)
            row["native_score"] = result.extra.get("native_score")
            row["native_metric"] = result.extra.get("native_metric", "zncc")
            row["_result"] = result
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            if raise_errors:
                raise
            rows.append({"method": name, "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc(), "hit": False})
    return rows

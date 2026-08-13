"""The adaptive coarse-to-fine pipeline. The whole point of the package.

Running wavelet + Fourier-Mellin + MACE + ZNCC on every image would be both
slow and dishonest: most images do not have a rotation problem, and most do
not have an ambiguity problem. Each stage here has an entry condition, and
the pipeline records which conditions fired, so a run explains itself:

    search image
        |
    preprocess (optional normalization)
        |
    wavelet pyramid  ------------------> candidate regions + ambiguity count
        |
    confidence gate --- high & unambiguous ----> straight to refinement
        |  low / ambiguous
    Fourier-Mellin (only if geometry looks wrong) --> corrected angle & size
        |
    MACE / OTDF (only if candidates are confusable) --> re-ranked shortlist
        |
    ZNCC refinement at full resolution on the shortlist
        |
    sub-pixel parabola fit -> final x, y

Every stage can be switched off through `PipelineConfig`, which is what makes
the ablation table (ZNCC / Wavelet+ZNCC / Wavelet+FM+ZNCC / Wavelet+MACE+ZNCC
/ full adaptive) a matter of configuration rather than of separate code paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import correlation_filters as cf
from .common import (
    DEFAULT_ANGLES,
    DEFAULT_SIZES,
    INNER_FRACTION,
    MatchResult,
    count_candidates,
    crop_roi,
    make_template,
    preprocess,
    top_k_peaks,
)
from .fourier_mellin import FM_ANGLE_LIMIT, FM_SCALE_LIMITS, estimate_rotation_scale
from .wavelet import wavelet_candidates
from .zncc import zncc_refine, zncc_surface


@dataclass
class PipelineConfig:
    """Stage switches and entry conditions. Defaults are the full adaptive run."""

    # preprocessing: none | clahe | zscore | bandpass
    preprocess_mode: str = "none"

    # stage 1 -- wavelet scout
    use_wavelet: bool = True
    wavelet_levels: int = 2
    top_k: int = 6

    # stage 2 -- confidence gate (skip everything clever if this passes)
    high_confidence: float = 0.55
    max_unambiguous_candidates: int = 2
    margin_ratio: float = 0.97          # 2nd/1st above this counts as a tie

    # stage 3 -- geometry, entered only on evidence of rotation/scale mismatch
    use_fourier_mellin: bool = True
    fm_angle_threshold: float = 1.5     # degrees
    fm_scale_threshold: float = 0.05    # fractional
    fm_min_confidence: float = 0.03

    # stage 4 -- discrimination, entered only when candidates are confusable
    use_mace: bool = True
    mace_alpha: float = 0.3             # 0 = MACE, >0 = OTDF trade-off
    mace_shortlist: int = 3
    # How far below the best ZNCC the discriminator's pick may sit before ZNCC
    # overrules it. This is the crux of the whole design: on a periodic array
    # the *wrong* lattice repeat routinely wins on ZNCC alone, so letting ZNCC
    # re-select would throw away exactly what the filter stage bought. A small
    # margin lets the filter decide between near-tied candidates while still
    # letting ZNCC veto a filter pick that is plainly not the pattern.
    zncc_veto_margin: float = 0.05

    # stage 5 -- final ZNCC refinement
    use_zncc_refine: bool = True
    refine_radius: int = 20
    sizes: tuple = DEFAULT_SIZES
    angles: tuple = DEFAULT_ANGLES
    always_run_all_stages: bool = False  # force every stage, for cost comparison

    name: str = "adaptive"


def _fallback_candidates(reference, search, top_k, sizes) -> dict:
    """Coarse candidates without the wavelet stage: one full-res ZNCC surface.

    Used when `use_wavelet` is off, so the ablation compares stages rather
    than comparing "has candidates" against "has none".
    """
    t0 = time.perf_counter()
    size = int(sizes[len(sizes) // 2])
    template = make_template(reference, size, 0.0, INNER_FRACTION)
    surface = zncc_surface(search, template)
    k = template.shape[0]
    peaks = top_k_peaks(surface, k=top_k, min_distance=max(k // 3, 4))
    _, best_score, _, _ = cv2.minMaxLoc(surface)
    return {
        "candidates": [(px + k / 2.0, py + k / 2.0, sc) for px, py, sc in peaks],
        "best_score": float(best_score),
        "n_candidates": count_candidates(surface, float(best_score)),
        "levels": 0,
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
    }


def adaptive_match(reference: np.ndarray, search: np.ndarray,
                   config: PipelineConfig = None) -> MatchResult:
    """Run the pipeline. The returned `stages` list is the decision trace."""
    cfg = config or PipelineConfig()
    t_start = time.perf_counter()
    stages = []

    def stage(name, action, t0, **info):
        stages.append({"stage": name, "action": action,
                       "ms": round((time.perf_counter() - t0) * 1000.0, 1), **info})

    # ---- stage 0: preprocessing -------------------------------------------
    t0 = time.perf_counter()
    ref = preprocess(reference, cfg.preprocess_mode)
    srch = preprocess(search, cfg.preprocess_mode)
    stage("preprocess", cfg.preprocess_mode, t0)

    # ---- stage 1: wavelet scout -------------------------------------------
    t0 = time.perf_counter()
    if cfg.use_wavelet:
        coarse = wavelet_candidates(ref, srch, levels=cfg.wavelet_levels, top_k=cfg.top_k)
        stage("wavelet", f"levels={cfg.wavelet_levels}", t0,
              candidates=len(coarse["candidates"]),
              best_score=round(coarse["best_score"], 4),
              rival_peaks=coarse["n_candidates"])
    else:
        coarse = _fallback_candidates(ref, srch, cfg.top_k, cfg.sizes)
        stage("wavelet", "skipped (full-res ZNCC surface instead)", t0,
              candidates=len(coarse["candidates"]),
              best_score=round(coarse["best_score"], 4),
              rival_peaks=coarse["n_candidates"])

    candidates = list(coarse["candidates"])
    scores = [c[2] for c in candidates]
    margin = (scores[1] / scores[0]) if len(scores) > 1 and scores[0] > 1e-6 else 0.0

    # ---- stage 2: confidence gate -----------------------------------------
    t0 = time.perf_counter()
    confident = (coarse["best_score"] >= cfg.high_confidence
                 and coarse["n_candidates"] <= cfg.max_unambiguous_candidates
                 and margin < cfg.margin_ratio)
    if cfg.always_run_all_stages:
        confident = False
    stage("confidence_gate", "high confidence -- skip FM and MACE" if confident
          else "low confidence or ambiguous -- continue", t0,
          best_score=round(coarse["best_score"], 4),
          rival_peaks=coarse["n_candidates"], margin=round(margin, 3))

    shortlist = candidates[:1] if confident else candidates
    base_angle = 0.0
    size_hint = None
    mace_ran = False

    # ---- stage 3: Fourier-Mellin, only on evidence of geometric mismatch ---
    if not confident and cfg.use_fourier_mellin:
        t0 = time.perf_counter()
        cx, cy, _ = candidates[0]
        base_template = make_template(ref, 100, 0.0, INNER_FRACTION)
        patch, _, _ = crop_roi(srch, cx, cy, int(base_template.shape[0] * 0.85))
        geom = estimate_rotation_scale(base_template, patch)
        angle, scale, conf = geom["angle"], geom["scale"], geom["confidence"]

        plausible = (abs(angle) <= FM_ANGLE_LIMIT
                     and FM_SCALE_LIMITS[0] <= scale <= FM_SCALE_LIMITS[1]
                     and conf >= cfg.fm_min_confidence)
        mismatched = abs(angle) > cfg.fm_angle_threshold or abs(scale - 1.0) > cfg.fm_scale_threshold

        if plausible and (mismatched or cfg.always_run_all_stages):
            base_angle = float(angle)
            size_hint = int(np.clip(round(100 * scale), min(cfg.sizes), max(cfg.sizes)))
            action = f"correcting angle {angle:+.1f} deg, scale {scale:.3f}"
        elif not plausible:
            action = "estimate not plausible -- ignored"
        else:
            action = "no geometric mismatch detected"
        stage("fourier_mellin", action, t0, angle=round(angle, 2),
              scale=round(scale, 3), confidence=round(conf, 4))
    elif not confident:
        stages.append({"stage": "fourier_mellin", "action": "disabled", "ms": 0.0})

    # ---- stage 4: MACE/OTDF discrimination, only when candidates are alike --
    ambiguous = (coarse["n_candidates"] > cfg.max_unambiguous_candidates
                 or margin >= cfg.margin_ratio)
    if not confident and cfg.use_mace and (ambiguous or cfg.always_run_all_stages):
        t0 = time.perf_counter()
        ranked = cf.rank_candidates(ref, srch, candidates, alpha=cfg.mace_alpha)
        if ranked:
            mace_ran = True
            shortlist = [(r["x"], r["y"], r["psr"]) for r in ranked[:cfg.mace_shortlist]]
            stage("mace", f"re-ranked {len(candidates)} candidates by PSR "
                          f"(alpha={cfg.mace_alpha:g})", t0,
                  top_psr=round(ranked[0]["psr"], 2),
                  runner_up_psr=round(ranked[1]["psr"], 2) if len(ranked) > 1 else None,
                  shortlist=len(shortlist))
        else:
            stage("mace", "no candidate could be scored", t0)
    elif not confident:
        stages.append({"stage": "mace", "action": "not needed -- candidates well separated"
                       if not ambiguous else "disabled", "ms": 0.0})

    # ---- stage 5: full-resolution ZNCC refinement + sub-pixel --------------
    t0 = time.perf_counter()
    sizes = (size_hint,) + tuple(cfg.sizes) if size_hint else tuple(cfg.sizes)
    best = None
    selection = "best ZNCC"
    if cfg.use_zncc_refine:
        refined_all = [
            zncc_refine(ref, srch, cx, cy, search_radius=cfg.refine_radius,
                        sizes=sizes, angles=cfg.angles, base_angle=base_angle)
            for cx, cy, _ in shortlist
        ]
        best_zncc = max(refined_all, key=lambda r: r.score)
        if mace_ran:
            # shortlist is in filter-PSR order, so refined_all[0] is the
            # discriminator's pick. It stands unless ZNCC vetoes it.
            top = refined_all[0]
            if top.score >= best_zncc.score - cfg.zncc_veto_margin:
                best, selection = top, "discriminator's pick accepted"
            else:
                best, selection = best_zncc, "discriminator's pick vetoed by ZNCC margin"
        else:
            best = best_zncc
        stage("zncc_refine",
              f"verified {len(shortlist)} candidate(s) at full resolution -- {selection}", t0,
              score=round(best.score, 4), size=best.size, angle=round(best.angle, 2))
    else:
        cx, cy, sc = shortlist[0]
        best = MatchResult(x=cx, y=cy, score=float(sc), method="candidate")
        stage("zncc_refine", "disabled -- returning the coarse candidate", t0)

    result = MatchResult(
        x=best.x, y=best.y, score=best.score, method=cfg.name,
        angle=best.angle, size=best.size,
        n_candidates=coarse["n_candidates"],
        elapsed_ms=(time.perf_counter() - t_start) * 1000.0,
        stages=stages,
        extra={
            "coarse_best_score": round(coarse["best_score"], 4),
            "candidate_margin": round(margin, 3),
            "took_fast_path": confident,
            "fm_base_angle": base_angle,
            "fm_size_hint": size_hint,
            "shortlist_size": len(shortlist),
            "selection": selection,
            "preprocess": cfg.preprocess_mode,
        },
    )
    return result


# ---------------------------------------------------------------------------
# named ablation configurations -- the rows of the comparison table
# ---------------------------------------------------------------------------

ABLATIONS = {
    "wavelet+zncc": PipelineConfig(
        name="wavelet+zncc", use_fourier_mellin=False, use_mace=False),
    "wavelet+fm+zncc": PipelineConfig(
        name="wavelet+fm+zncc", use_fourier_mellin=True, use_mace=False,
        high_confidence=2.0),          # unreachable => always exercise FM
    "wavelet+mace+zncc": PipelineConfig(
        name="wavelet+mace+zncc", use_fourier_mellin=False, use_mace=True,
        mace_alpha=0.0,                # pure MACE, to contrast with the default
        high_confidence=2.0),          # unreachable => always exercise MACE
    "adaptive": PipelineConfig(name="adaptive"),
    "adaptive_all_stages": PipelineConfig(
        name="adaptive_all_stages", always_run_all_stages=True),
}


def explain(result: MatchResult) -> str:
    """One line per stage: what ran, what it decided, what it cost."""
    lines = []
    for s in result.stages:
        detail = ", ".join(f"{k}={v}" for k, v in s.items()
                           if k not in ("stage", "action", "ms") and v is not None)
        line = f"{s['stage']:<16} {s['ms']:>7.1f} ms  {s['action']}"
        if detail:
            line += f"   [{detail}]"
        lines.append(line)
    return "\n".join(lines)

import numpy as np
import cv2
import pywt
from scipy.fft import fft2, ifft2, fftshift
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================
# STAGE 1: OPTIONAL / MILD WAVELET DENOISING
# ============================================================

def wavelet_denoise(
    img: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
    sigma: Optional[float] = None,
    threshold_scale: float = 0.35,
) -> np.ndarray:
    """
    Mild wavelet denoising.

    threshold_scale < 1 keeps fine texture that may be important
    for repetitive semiconductor-pattern matching.
    """

    img = img.astype(np.float64)

    coeffs = pywt.wavedec2(
        img,
        wavelet=wavelet,
        level=level
    )

    cA, detail_levels = coeffs[0], coeffs[1:]

    if sigma is None:
        finest_cD = detail_levels[-1][2]
        sigma = np.median(np.abs(finest_cD)) / 0.6745

    n = img.size

    thresh = (
        threshold_scale
        * sigma
        * np.sqrt(2 * np.log(max(n, 2)))
    )

    denoised_details = []

    for cH, cV, cD in detail_levels:

        denoised_details.append(
            (
                pywt.threshold(
                    cH,
                    thresh,
                    mode="soft"
                ),

                pywt.threshold(
                    cV,
                    thresh,
                    mode="soft"
                ),

                pywt.threshold(
                    cD,
                    thresh,
                    mode="soft"
                )
            )
        )

    out = pywt.waverec2(
        [cA] + denoised_details,
        wavelet=wavelet
    )

    return out[
        :img.shape[0],
        :img.shape[1]
    ].astype(np.float32)


# ============================================================
# STAGE 2: LATTICE / PERIODICITY ESTIMATION
# ============================================================

def estimate_lattice_pitch(
    img: np.ndarray
) -> Tuple[float, float]:
    """
    Estimate dominant x/y periodicity using autocorrelation.
    """

    img = img.astype(np.float64)

    img = img - np.mean(img)

    F = fft2(img)

    power = np.abs(F) ** 2

    autocorr = np.real(
        ifft2(power)
    )

    autocorr = fftshift(
        autocorr
    )

    h, w = autocorr.shape

    cy = h // 2
    cx = w // 2

    def first_peak(line):

        if len(line) < 5:
            return 0.0

        center_val = max(
            abs(autocorr[cy, cx]),
            1e-12
        )

        line = line / center_val

        for i in range(4, len(line) - 1):

            if (
                line[i] > line[i - 1]
                and
                line[i] >= line[i + 1]
                and
                line[i] > 0.10
            ):
                return float(i)

        return 0.0

    pitch_x = first_peak(
        autocorr[cy, cx:]
    )

    pitch_y = first_peak(
        autocorr[cy:, cx]
    )

    return (
        pitch_x,
        pitch_y
    )


def scale_from_lattice_pitch(
    template_pitch: Tuple[float, float],
    search_pitch: Tuple[float, float],
) -> Optional[float]:
    """
    Estimate template -> search scale from lattice pitch.
    """

    tx, ty = template_pitch

    sx, sy = search_pitch

    ratios = []

    if tx > 0 and sx > 0:
        ratios.append(
            sx / tx
        )

    if ty > 0 and sy > 0:
        ratios.append(
            sy / ty
        )

    if not ratios:
        return None

    return float(
        np.median(ratios)
    )


def generate_scales(
    center_scale: Optional[float],
    min_scale: float = 0.20,
    max_scale: float = 0.50,
    step: float = 0.01,
) -> List[float]:
    """
    Generate candidate template scales.

    If lattice pitch gives a scale estimate,
    search densely around it.
    """

    if (
        center_scale is not None
        and
        0.05 < center_scale < 2.0
    ):

        lo = max(
            min_scale,
            center_scale - 0.12
        )

        hi = min(
            max_scale,
            center_scale + 0.12
        )

        scales = np.arange(
            lo,
            hi + step * 0.5,
            step
        ).tolist()

        scales.append(
            center_scale
        )

        scales = sorted(
            set(
                round(
                    float(s),
                    4
                )
                for s in scales
            )
        )

        return scales

    return [
        round(
            float(s),
            4
        )
        for s in np.arange(
            min_scale,
            max_scale + step * 0.5,
            step
        )
    ]


# ============================================================
# CANDIDATE DATA STRUCTURE
# ============================================================

@dataclass
class Candidate:

    x: int

    y: int

    scale: float

    score: float

    psr: float = 0.0

    stability: float = 0.0


# ============================================================
# PSR
# ============================================================

def peak_psr(
    result: np.ndarray,
    peak_loc: Tuple[int, int],
    radius: int = 5,
) -> float:
    """
    Peak-to-sidelobe ratio.
    """

    x, y = peak_loc

    y0 = max(
        0,
        y - radius
    )

    y1 = min(
        result.shape[0],
        y + radius + 1
    )

    x0 = max(
        0,
        x - radius
    )

    x1 = min(
        result.shape[1],
        x + radius + 1
    )

    mask = np.ones(
        result.shape,
        dtype=bool
    )

    mask[
        y0:y1,
        x0:x1
    ] = False

    sidelobes = result[mask]

    if sidelobes.size == 0:
        return 0.0

    return float(
        (
            result[y, x]
            - np.mean(sidelobes)
        )
        /
        (
            np.std(sidelobes)
            + 1e-8
        )
    )


# ============================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================

def nms_candidates(
    candidates: List[Candidate],
    radius: int,
    top_k: int,
) -> List[Candidate]:

    candidates = sorted(
        candidates,
        key=lambda c: c.score,
        reverse=True
    )

    selected = []

    for cand in candidates:

        too_close = False

        for chosen in selected:

            dx = (
                cand.x
                - chosen.x
            )

            dy = (
                cand.y
                - chosen.y
            )

            if (
                dx * dx
                +
                dy * dy
                <
                radius * radius
            ):

                too_close = True

                break

        if not too_close:

            selected.append(
                cand
            )

        if len(selected) >= top_k:
            break

    return selected


# ============================================================
# MULTI-SCALE NCC
# ============================================================

def multiscale_candidates(
    search_img: np.ndarray,
    template: np.ndarray,
    scales: List[float],
    top_k_per_scale: int = 3,
    final_top_k: int = 12,
) -> List[Candidate]:
    """
    Scale-aware coarse localization.

    The template is explicitly resized at every candidate scale.

    This is the main fix for the ~3.3x scale mismatch.
    """

    search = search_img.astype(
        np.float32
    )

    templ = template.astype(
        np.float32
    )

    all_candidates = []

    for scale in scales:

        new_w = max(
            8,
            int(
                round(
                    templ.shape[1]
                    * scale
                )
            )
        )

        new_h = max(
            8,
            int(
                round(
                    templ.shape[0]
                    * scale
                )
            )
        )

        if (
            new_w >= search.shape[1]
            or
            new_h >= search.shape[0]
        ):
            continue

        scaled_template = cv2.resize(
            templ,
            (
                new_w,
                new_h
            ),
            interpolation=(
                cv2.INTER_AREA
                if scale < 1.0
                else cv2.INTER_CUBIC
            )
        )

        result = cv2.matchTemplate(
            search,
            scaled_template,
            cv2.TM_CCOEFF_NORMED
        )

        work = result.copy()

        radius = max(
            4,
            min(
                new_w,
                new_h
            ) // 3
        )

        for _ in range(
            top_k_per_scale
        ):

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(work)
            if not np.isfinite(
                max_val
            ):
                break

            x_top, y_top = max_loc

            # IMPORTANT:
            # cv2.matchTemplate returns
            # TOP-LEFT coordinates.
            #
            # Convert them to CENTER coordinates.

            cx = (
                x_top
                +
                new_w // 2
            )

            cy = (
                y_top
                +
                new_h // 2
            )

            psr = peak_psr(
                result,
                (
                    x_top,
                    y_top
                )
            )

            all_candidates.append(
                Candidate(
                    x=int(cx),
                    y=int(cy),
                    scale=float(scale),
                    score=float(max_val),
                    psr=float(psr),
                )
            )

            x0 = max(
                0,
                x_top - radius
            )

            x1 = min(
                work.shape[1],
                x_top + radius + 1
            )

            y0 = max(
                0,
                y_top - radius
            )

            y1 = min(
                work.shape[0],
                y_top + radius + 1
            )

            work[
                y0:y1,
                x0:x1
            ] = -1.0

    if all_candidates:

        median_size = int(
            np.median(
                [
                    min(
                        template.shape[0]
                        * c.scale,

                        template.shape[1]
                        * c.scale
                    )

                    for c in all_candidates
                ]
            )
        )

        nms_radius = max(
            8,
            median_size // 2
        )

    else:

        nms_radius = 8

    return nms_candidates(
        all_candidates,
        nms_radius,
        final_top_k
    )


# ============================================================
# SCALE STABILITY
# ============================================================

def evaluate_scale_stability(
    search_img: np.ndarray,
    template: np.ndarray,
    candidate: Candidate,
    delta: float = 0.01,
) -> Tuple[float, float]:
    """
    Re-run NCC around the selected scale.

    Returns:

        score_std
        location_std
    """

    scales = [
        max(
            0.05,
            candidate.scale - delta
        ),

        candidate.scale,

        candidate.scale + delta
    ]

    locations = []

    scores = []

    for scale in scales:

        tw = max(
            8,
            int(
                round(
                    template.shape[1]
                    * scale
                )
            )
        )

        th = max(
            8,
            int(
                round(
                    template.shape[0]
                    * scale
                )
            )
        )

        if (
            tw >= search_img.shape[1]
            or
            th >= search_img.shape[0]
        ):
            continue

        st = cv2.resize(
            template.astype(
                np.float32
            ),
            (
                tw,
                th
            ),
            interpolation=(
                cv2.INTER_AREA
                if scale < 1
                else cv2.INTER_CUBIC
            )
        )

        result = cv2.matchTemplate(
            search_img.astype(
                np.float32
            ),
            st,
            cv2.TM_CCOEFF_NORMED
        )

        min_val, score, min_loc, loc = cv2.minMaxLoc(result)

        cx = (
            loc[0]
            +
            tw // 2
        )

        cy = (
            loc[1]
            +
            th // 2
        )

        locations.append(
            (
                cx,
                cy
            )
        )

        scores.append(
            score
        )

    if not scores:

        return (
            999.0,
            999.0
        )

    score_std = float(
        np.std(scores)
    )

    if len(locations) > 1:

        location_std = float(
            np.mean(
                [
                    np.sqrt(
                        (
                            x
                            -
                            candidate.x
                        ) ** 2
                        +
                        (
                            y
                            -
                            candidate.y
                        ) ** 2
                    )

                    for x, y
                    in locations
                ]
            )
        )

    else:

        location_std = 999.0

    return (
        score_std,
        location_std
    )


# ============================================================
# MACE / OTSDF FILTER
# ============================================================

@dataclass
class CompositeFilter:

    H: np.ndarray

    shape: Tuple[int, int]


def synthesize_mace(
    training_images: List[np.ndarray],
    alpha: float = 0.15,
) -> CompositeFilter:
    """
    MACE / OTSDF filter synthesis.
    """

    shape = training_images[0].shape

    N = (
        shape[0]
        *
        shape[1]
    )

    n_train = len(
        training_images
    )

    X = np.zeros(
        (
            N,
            n_train
        ),
        dtype=complex
    )

    for i, img in enumerate(
        training_images
    ):

        F = fft2(
            img.astype(
                np.float64
            )
        )

        X[:, i] = F.flatten()

    power_spectrum = np.mean(
        np.abs(X) ** 2,
        axis=1
    )

    D_diag = (
        (1.0 - alpha)
        *
        power_spectrum
        +
        alpha
        *
        np.mean(
            power_spectrum
        )
    )

    D_diag = np.maximum(
        D_diag,
        1e-8
    )

    D_inv = 1.0 / D_diag

    u = np.ones(
        (
            n_train,
            1
        ),
        dtype=complex
    )

    D_inv_X = (
        X
        *
        D_inv[:, None]
    )

    XhDinvX = (
        X.conj().T
        @
        D_inv_X
    )

    XhDinvX += (
        1e-8
        *
        np.eye(n_train)
    )

    coeffs = np.linalg.solve(
        XhDinvX,
        u
    )

    H = (
        D_inv_X
        @
        coeffs
    ).flatten()

    H = H.reshape(
        shape
    )

    return CompositeFilter(
        H=H,
        shape=shape
    )


def make_training_set(
    template: np.ndarray,
    count: int = 6,
) -> List[np.ndarray]:
    """
    Generate shifted/noisy template variants.
    """

    template = template.astype(
        np.float32
    )

    rng = np.random.default_rng(
        0
    )

    std = max(
        float(
            np.std(template)
        ),
        1e-6
    )

    training = [
        template.copy()
    ]

    for _ in range(
        count - 1
    ):

        noisy = (
            template
            +
            rng.normal(
                0,
                std * 0.02,
                template.shape
            ).astype(
                np.float32
            )
        )

        dy = int(
            rng.integers(
                -2,
                3
            )
        )

        dx = int(
            rng.integers(
                -2,
                3
            )
        )

        shifted = np.roll(
            noisy,
            shift=(
                dy,
                dx
            ),
            axis=(
                0,
                1
            )
        )

        training.append(
            shifted
        )

    return training


# ============================================================
# LOCAL NCC REFINEMENT
# ============================================================

def local_refine(
    search_img: np.ndarray,
    template: np.ndarray,
    candidate: Candidate,
    margin_fraction: float = 0.40,
) -> Candidate:
    """
    Refine candidate location using NCC around the candidate.
    """

    scale = candidate.scale

    tw = max(
        8,
        int(
            round(
                template.shape[1]
                *
                scale
            )
        )
    )

    th = max(
        8,
        int(
            round(
                template.shape[0]
                *
                scale
            )
        )
    )

    scaled_template = cv2.resize(
        template.astype(
            np.float32
        ),
        (
            tw,
            th
        ),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1
            else cv2.INTER_CUBIC
        )
    )

    margin_x = max(
        8,
        int(
            tw
            *
            margin_fraction
        )
    )

    margin_y = max(
        8,
        int(
            th
            *
            margin_fraction
        )
    )

    x0 = max(
        0,
        candidate.x
        -
        tw // 2
        -
        margin_x
    )

    y0 = max(
        0,
        candidate.y
        -
        th // 2
        -
        margin_y
    )

    x1 = min(
        search_img.shape[1],
        candidate.x
        +
        tw // 2
        +
        margin_x
    )

    y1 = min(
        search_img.shape[0],
        candidate.y
        +
        th // 2
        +
        margin_y
    )

    window = search_img[
        y0:y1,
        x0:x1
    ]

    if (
        window.shape[0] < th
        or
        window.shape[1] < tw
    ):

        return candidate

    result = cv2.matchTemplate(
        window.astype(
            np.float32
        ),
        scaled_template,
        cv2.TM_CCOEFF_NORMED
    )

    min_val, score, min_loc, loc = cv2.minMaxLoc(result)

    new_x = (
        x0
        +
        loc[0]
        +
        tw // 2
    )

    new_y = (
        y0
        +
        loc[1]
        +
        th // 2
    )

    psr = peak_psr(
        result,
        loc
    )

    return Candidate(
        x=int(new_x),
        y=int(new_y),
        scale=scale,
        score=float(score),
        psr=float(psr),
        stability=candidate.stability
    )


# ============================================================
# LATTICE DISAMBIGUATION
# ============================================================

def lattice_distance(
    dx: float,
    dy: float,
    pitch: Tuple[float, float],
) -> float:
    """
    Distance of offset from nearest lattice multiple.
    """

    px, py = pitch

    terms = []

    if px > 0:

        nx = round(
            dx / px
        )

        terms.append(
            abs(
                dx
                -
                nx * px
            )
        )

    if py > 0:

        ny = round(
            dy / py
        )

        terms.append(
            abs(
                dy
                -
                ny * py
            )
        )

    if not terms:

        return float(
            "inf"
        )

    return float(
        min(terms)
    )


def disambiguate_candidates(
    candidates: List[Candidate],
    pitch: Tuple[float, float],
    score_ratio_thresh: float = 1.03,
    pitch_tolerance_px: float = 2.5,
) -> Optional[Candidate]:
    """
    Reject only when two candidates are almost equally good and
    their separation is consistent with the repetitive lattice.
    """

    if not candidates:

        return None

    candidates = sorted(
        candidates,
        key=lambda c: c.score,
        reverse=True
    )

    best = candidates[0]

    for other in candidates[1:]:

        dx = (
            other.x
            -
            best.x
        )

        dy = (
            other.y
            -
            best.y
        )

        ratio = (
            best.score
            /
            (
                other.score
                +
                1e-8
            )
        )

        lattice_err = lattice_distance(
            dx,
            dy,
            pitch
        )

        if (
            ratio < score_ratio_thresh
            and
            lattice_err
            <
            pitch_tolerance_px
        ):

            return None

    return best


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def find_template(
    search_img: np.ndarray,
    template: np.ndarray,
    use_denoising: bool = False,
    use_fourier_mellin: bool = False,
    mace_alpha: float = 0.15,
    synthetic_variants: int = 6,
) -> dict:
    """
    Scale-aware template matching pipeline.

    Pipeline:

        optional denoising
              ↓
        lattice pitch
              ↓
        scale estimation
              ↓
        multi-scale NCC
              ↓
        stability analysis
              ↓
        local refinement
              ↓
        lattice disambiguation
    """

    search = search_img.astype(
        np.float32
    )

    templ = template.astype(
        np.float32
    )

    # --------------------------------------------------------
    # 1. OPTIONAL DENOISING
    # --------------------------------------------------------

    if use_denoising:

        search_proc = wavelet_denoise(
            search,
            threshold_scale=0.35
        )

        templ_proc = wavelet_denoise(
            templ,
            threshold_scale=0.35
        )

    else:

        # IMPORTANT:
        # Preserve the fine dot-pattern texture.
        search_proc = search.copy()
        templ_proc = templ.copy()

    # --------------------------------------------------------
    # 2. LATTICE PITCH
    # --------------------------------------------------------

    template_pitch = estimate_lattice_pitch(
        templ_proc
    )

    search_pitch = estimate_lattice_pitch(
        search_proc
    )

    lattice_scale = scale_from_lattice_pitch(
        template_pitch,
        search_pitch
    )

    # --------------------------------------------------------
    # 3. GENERATE SCALE RANGE
    # --------------------------------------------------------

    scales = generate_scales(
        lattice_scale,
        min_scale=0.20,
        max_scale=0.50,
        step=0.01
    )

    print()
    print(
        "========== SCALE ANALYSIS =========="
    )

    print(
        f"Template pitch : {template_pitch}"
    )

    print(
        f"Search pitch   : {search_pitch}"
    )

    print(
        f"Estimated scale: {lattice_scale}"
    )

    print(
        f"Scales tested  : {len(scales)}"
    )

    print(
        "===================================="
    )

    print()

    # --------------------------------------------------------
    # 4. MULTI-SCALE NCC
    # --------------------------------------------------------

    candidates = multiscale_candidates(
        search_proc,
        templ_proc,
        scales,
        top_k_per_scale=3,
        final_top_k=12
    )

    if not candidates:

        return {
            "result": None,
            "all_candidates": [],
            "estimated_scale": lattice_scale,
            "template_pitch": template_pitch,
            "search_pitch": search_pitch,
            "scales_tested": scales,
        }

    # --------------------------------------------------------
    # 5. STABILITY
    # --------------------------------------------------------

    stable_candidates = []

    print(
        "========== CANDIDATES =========="
    )

    for cand in candidates:

        score_std, location_std = (
            evaluate_scale_stability(
                search_proc,
                templ_proc,
                cand,
                delta=0.01
            )
        )

        cand.stability = (
            location_std
        )

        print(
            f"x={cand.x:4d} "
            f"y={cand.y:4d} "
            f"scale={cand.scale:.3f} "
            f"NCC={cand.score:.4f} "
            f"PSR={cand.psr:.2f} "
            f"loc_stability={location_std:.2f}px"
        )

        # Reject wildly unstable candidates.
        if location_std <= 8.0:

            stable_candidates.append(
                cand
            )

    # If everything is unstable, keep the best few
    # rather than returning nothing.
    if not stable_candidates:

        stable_candidates = sorted(
            candidates,
            key=lambda c: (
                c.stability,
                -c.psr,
                -c.score
            )
        )[:5]

    # --------------------------------------------------------
    # 6. LOCAL REFINEMENT
    # --------------------------------------------------------

    refined = []

    for cand in stable_candidates:

        r = local_refine(
            search_proc,
            templ_proc,
            cand,
            margin_fraction=0.40
        )

        refined.append(
            r
        )

    # --------------------------------------------------------
    # 7. FINAL RANKING
    # --------------------------------------------------------

    def quality(c: Candidate) -> float:

        stability_penalty = min(
            c.stability / 50.0,
            0.20
        )

        return (
            0.65 * c.score
            +
            0.25
            *
            np.tanh(
                c.psr / 10.0
            )
            +
            0.10
            *
            np.exp(
                -c.stability / 10.0
            )
            -
            stability_penalty
        )

    refined = sorted(
        refined,
        key=quality,
        reverse=True
    )

    # --------------------------------------------------------
    # 8. LATTICE DISAMBIGUATION
    # --------------------------------------------------------

    result = disambiguate_candidates(
        refined,
        search_pitch,
        score_ratio_thresh=1.03,
        pitch_tolerance_px=2.5
    )

    return {
        "result": result,
        "all_candidates": refined,
        "estimated_scale": lattice_scale,
        "template_pitch": template_pitch,
        "search_pitch": search_pitch,
        "scales_tested": scales,
    }


# ============================================================
# VISUALIZATION
# ============================================================

def draw_results(
    search_img: np.ndarray,
    template: np.ndarray,
    candidates: List[Candidate],
    result: Optional[Candidate],
    filename: str = "match_result.png",
):
    """
    Draw boxes using the ACTUAL detected template scale.
    """

    vis = cv2.cvtColor(
        search_img.astype(
            np.uint8
        ),
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # ALL CANDIDATES
    # --------------------------------------------------------

    for cand in candidates:

        tw = max(
            8,
            int(
                round(
                    template.shape[1]
                    *
                    cand.scale
                )
            )
        )

        th = max(
            8,
            int(
                round(
                    template.shape[0]
                    *
                    cand.scale
                )
            )
        )

        x0 = (
            cand.x
            -
            tw // 2
        )

        y0 = (
            cand.y
            -
            th // 2
        )

        x1 = (
            cand.x
            +
            tw // 2
        )

        y1 = (
            cand.y
            +
            th // 2
        )

        cv2.rectangle(
            vis,
            (x0, y0),
            (x1, y1),
            (0, 165, 255),
            1
        )

        cv2.putText(
            vis,
            f"{cand.scale:.2f}:{cand.score:.2f}",
            (
                x0,
                max(
                    12,
                    y0 - 3
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 165, 255),
            1
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if result is not None:

        tw = max(
            8,
            int(
                round(
                    template.shape[1]
                    *
                    result.scale
                )
            )
        )

        th = max(
            8,
            int(
                round(
                    template.shape[0]
                    *
                    result.scale
                )
            )
        )

        x0 = (
            result.x
            -
            tw // 2
        )

        y0 = (
            result.y
            -
            th // 2
        )

        x1 = (
            result.x
            +
            tw // 2
        )

        y1 = (
            result.y
            +
            th // 2
        )

        cv2.rectangle(
            vis,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            2
        )

        cv2.putText(
            vis,
            (
                f"BEST "
                f"scale={result.scale:.3f} "
                f"NCC={result.score:.3f} "
                f"PSR={result.psr:.1f}"
            ),
            (
                max(5, x0),
                max(20, y0 - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        cv2.circle(
            vis,
            (
                result.x,
                result.y
            ),
            4,
            (0, 255, 0),
            -1
        )

    cv2.imwrite(
        filename,
        vis
    )

    return vis


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    search = cv2.imread(
        "big_image.png",
        cv2.IMREAD_GRAYSCALE
    )

    template = cv2.imread(
        "small_part.png",
        cv2.IMREAD_GRAYSCALE
    )

    if search is None:

        raise FileNotFoundError(
            "Could not load big_image.png"
        )

    if template is None:

        raise FileNotFoundError(
            "Could not load small_part.png"
        )

    print()
    print(
        "=============================================="
    )

    print(
        " SCALE-AWARE REPETITIVE-PATTERN MATCHING"
    )

    print(
        "=============================================="
    )

    out = find_template(
        search,
        template,

        # Start WITHOUT denoising.
        # Your dot pattern is the signal.
        use_denoising=False,

        # Do not use Fourier-Mellin for the
        # primary scale correction.
        use_fourier_mellin=False,

        mace_alpha=0.15,

        synthetic_variants=6
    )

    print()
    print(
        "========== FINAL RESULT =========="
    )

    result = out["result"]

    if result is None:

        print(
            "RESULT: AMBIGUOUS / NO TRUSTWORTHY MATCH"
        )

    else:

        print(
            f"Best match:"
            f" x={result.x},"
            f" y={result.y},"
            f" scale={result.scale:.4f},"
            f" NCC={result.score:.4f},"
            f" PSR={result.psr:.2f},"
            f" stability={result.stability:.2f}px"
        )

    print(
        f"Template pitch = "
        f"{out['template_pitch']}"
    )

    print(
        f"Search pitch   = "
        f"{out['search_pitch']}"
    )

    print(
        f"Estimated scale = "
        f"{out['estimated_scale']}"
    )

    # --------------------------------------------------------
    # SAVE VISUALIZATION
    # --------------------------------------------------------

    draw_results(
        search,
        template,
        out["all_candidates"],
        result,
        filename="match_result.png"
    )

    print()
    print(
        "Saved: match_result.png"
    )
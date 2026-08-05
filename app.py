"""
Drift-Sense synthetic data explorer -- Streamlit app for Hugging Face Spaces.

This is a thin UI shell around the real generator in `src/`: every image
shown here comes from the exact same `generate_sample()` pipeline students
use from the CLI, so nothing here is a separate/approximated reimplementation.
"""

import io
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import GenerationParams, generate_sample
from src.presets import PRESETS, get_preset

st.set_page_config(page_title="Drift-Sense Synthetic Data Explorer", layout="wide")
st.title("Drift-Sense: Synthetic Dataset Explorer")
st.caption(
    "Reference: 1000x1000 px @ 1 nm/px (1 um FOV). "
    "Search: 1000x1000 px @ 10 nm/px (10 um FOV). "
    "The reference's footprint in the search image is 100x100 px -- the '10x shrink' "
    "in the problem statement falls directly out of that pixel-size ratio."
)

if "seed" not in st.session_state:
    st.session_state.seed = 42

with st.sidebar:
    st.header("Structure")
    architecture = st.selectbox("Architecture preset", list(PRESETS.keys()))
    feature_scale = st.slider(
        "Feature size scale", 0.5, 2.0, 1.0, 0.05,
        help="Scales every pitch/width in the preset proportionally -- explore other process nodes.",
    )

    st.header("SEM imaging physics")
    beam_spot_size_nm = st.slider(
        "Beam spot size (nm)", 1.0, 20.0, 5.0, 0.5,
        help="Single physical PSF blur applied before downsampling. Larger spot = "
             "more smearing of dense structures once downsampled into the search image.",
    )
    collapse_threshold_nm = st.slider(
        "Pattern-collapse threshold (nm)", 0.0, 20.0, 10.0, 1.0,
        help="Gaps/lines below this (structural defect, not imaging) probabilistically bridge. "
             "Default 10 nm = exactly 1 px at the search image's 10 nm/px resolution.",
    )

    st.header("Acquisition noise")
    dose_reference = st.slider("Reference dose (higher = cleaner)", 100.0, 5000.0, 2000.0, 100.0)
    dose_search = st.slider("Search dose (higher = cleaner)", 20.0, 2000.0, 200.0, 20.0)
    shear_amplitude_px = st.slider("Search raster drift/shear (px)", 0.0, 5.0, 1.5, 0.1)
    drift_jitter_px = st.slider("Search row jitter (px)", 0.0, 3.0, 0.5, 0.1)

    st.header("Distortion & polygon scaling")
    linewidth_bias_nm = st.slider(
        "Linewidth/CD bias (nm)", -10.0, 10.0, 0.0, 0.5,
        help="Deterministic global over/under-exposure bias applied to every drawn feature.",
    )
    corner_rounding_px = st.slider(
        "Corner rounding (px)", 0.0, 6.0, 0.0, 0.5,
        help="Morphological rounding of polygon corners -- real litho/etch never draws sharp corners.",
    )
    astigmatism_ratio = st.slider(
        "Beam astigmatism ratio", 0.5, 2.0, 1.0, 0.05,
        help="Elliptical beam spot (sigmaY = sigmaX * ratio) -- directional blur.",
    )
    barrel_distortion_k = st.slider(
        "Barrel(+)/pincushion(-) distortion", -0.15, 0.15, 0.0, 0.01,
        help="Radial lens-style distortion from imperfect beam-scan linearity.",
    )
    vignette_strength = st.slider("Vignette strength", 0.0, 1.0, 0.0, 0.05)
    gamma = st.slider("Gamma (contrast curve)", 0.4, 2.5, 1.0, 0.05)
    charging_streak_prob = st.slider(
        "Charging streaks (per 100 rows)", 0.0, 5.0, 0.0, 0.25,
        help="Bright horizontal streaks from local sample charging on insulating regions.",
    )
    charging_streak_intensity = st.slider("Charging streak intensity", 0.0, 3.0, 0.0, 0.1)

    st.header("Die layout (multi-region)")
    mat_size_nm = st.slider(
        "Array block (mat) size (nm)", 800.0, 5000.0, 2600.0, 100.0,
        help="Size of each independently-generated structure block before a separating strip.",
    )
    strip_width_nm = st.slider(
        "Separator strip width (nm)", 80.0, 800.0, 320.0, 20.0,
        help="Width of the peripheral/routing material band between array blocks.",
    )
    boundary_bias = st.slider(
        "P(reference crop straddles a boundary)", 0.0, 1.0, 0.35, 0.05,
        help="Probability the Reference crop is deliberately placed across a mat/strip edge "
             "instead of sampled uniformly at random.",
    )

    st.header(" ")
    show_gt_box = st.checkbox("Show ground-truth box", value=True)
    if st.button("Regenerate (new seed)"):
        st.session_state.seed = np.random.randint(0, 2_000_000_000)


def build_params():
    return GenerationParams(
        beam_spot_size_nm=beam_spot_size_nm,
        collapse_threshold_nm=collapse_threshold_nm,
        dose_reference=dose_reference,
        dose_search=dose_search,
        shear_amplitude_px=shear_amplitude_px,
        drift_jitter_px=drift_jitter_px,
        linewidth_bias_nm=linewidth_bias_nm,
        corner_rounding_px=corner_rounding_px,
        astigmatism_ratio=astigmatism_ratio,
        barrel_distortion_k=barrel_distortion_k,
        vignette_strength=vignette_strength,
        gamma=gamma,
        charging_streak_prob=charging_streak_prob,
        charging_streak_intensity=charging_streak_intensity,
        mat_size_nm=mat_size_nm,
        strip_width_nm=strip_width_nm,
        boundary_bias=boundary_bias,
    )


preset_overrides = None
if feature_scale != 1.0:
    base_preset = get_preset(architecture)
    preset_overrides = {
        k: v * feature_scale for k, v in base_preset.items() if k.endswith("_nm")
    }
    st.sidebar.caption(f"Scaled preset: {preset_overrides}")

rng = np.random.default_rng(st.session_state.seed)
params = build_params()
sample = generate_sample(architecture, rng, params, preset_overrides)

search_display = cv2.cvtColor(sample["search_img"], cv2.COLOR_GRAY2BGR)
if show_gt_box:
    x0, y0, w, h = sample["gt_box"]
    cv2.rectangle(
        search_display,
        (int(round(x0)), int(round(y0))),
        (int(round(x0 + w)), int(round(y0 + h))),
        (0, 0, 255),
        2,
    )

col1, col2 = st.columns(2)
with col1:
    st.subheader("Reference (1 nm/px)")
    st.image(sample["reference_img"], clamp=True, use_container_width=True)
with col2:
    st.subheader("Search (10 nm/px)")
    st.image(cv2.cvtColor(search_display, cv2.COLOR_BGR2RGB), use_container_width=True)

st.markdown(
    f"**Ground truth center in search image:** `({sample['gt_x']:.1f}, {sample['gt_y']:.1f})` px "
    f"&nbsp;&nbsp;|&nbsp;&nbsp; **Architecture:** `{architecture}` &nbsp;&nbsp;|&nbsp;&nbsp; **Seed:** `{st.session_state.seed}`"
)


def to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button("Download reference.png", to_png_bytes(sample["reference_img"]), "reference.png", "image/png")
with dl2:
    st.download_button("Download search.png", to_png_bytes(sample["search_img"]), "search.png", "image/png")
with dl3:
    metadata = {
        "architecture": architecture,
        "gt_x": sample["gt_x"],
        "gt_y": sample["gt_y"],
        "gt_box": sample["gt_box"],
        "seed": st.session_state.seed,
        **sample["params"],
    }
    st.download_button("Download metadata.json", json.dumps(metadata, indent=2), "metadata.json", "application/json")

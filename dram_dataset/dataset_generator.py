"""
Adapter exposing the `dram_synth` pipeline in the shape `evaluate.py` expects.

`evaluate.py` wants two things from this module:

    generate_pair(style=..., seed=...) -> (reference, wide, gt)
    to_rgb_variant(img, rng)           -> 3-channel variant of a grayscale image

`generate_pair` builds a fresh sample on demand rather than reading the saved
dataset, so the harness can run any number of randomized cases without a
dataset having been generated first. Each seed is self-contained and
reproducible: the same seed always yields the same pair and the same ground
truth.

Ground truth is returned as an object with `.center_x` / `.center_y` in wide
(search) image pixels, which is what the harness scores against. Those
coordinates are computed analytically inside `dram_synth.sample`, never
measured from the rendered image.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from dram_synth.params import DIFFICULTIES, DIFFICULTY_MIX
from dram_synth.sample import build_sample

# This deliverable is DRAM-only, per the dataset specification.
SUPPORTED_STYLES = ("dram",)


@dataclass
class GroundTruth:
    """True location of the reference inside the wide/search image."""
    center_x: float
    center_y: float
    x1: float
    y1: float
    x2: float
    y2: float
    footprint_px: float
    rotation_deg: float
    scale: float
    difficulty: str
    seed: int
    quad: list = field(default_factory=list)

    @property
    def bbox(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


def difficulty_for_seed(seed: int) -> str:
    """Pick a difficulty tier for a bare seed, respecting the 25/50/25 mix."""
    rng = np.random.default_rng((int(seed) * 2654435761) % (2 ** 32))
    probs = [DIFFICULTY_MIX[d] for d in DIFFICULTIES]
    return str(rng.choice(list(DIFFICULTIES), p=probs))


def generate_pair(style: str = "dram", seed: int = 0,
                  difficulty: str | None = None, supersample: int = 10):
    """Generate one (reference, wide, ground_truth) triple.

    `reference` is a small high-magnification grayscale image (100-256 px
    square); `wide` is the 1000x1000 low-magnification search image. Both are
    uint8 and independently degraded -- separate noise, blur, drift, vibration
    and contrast, drawn from their own random streams.
    """
    if style not in SUPPORTED_STYLES:
        raise ValueError(
            f"style {style!r} is not available: this generator is DRAM-only "
            f"(supported: {list(SUPPORTED_STYLES)}). Run the harness with "
            f"--styles dram."
        )

    difficulty = difficulty or difficulty_for_seed(seed)
    sample = build_sample(int(seed), difficulty, supersample)
    b = sample["bbox"]
    p = sample["params"]

    gt = GroundTruth(
        center_x=sample["center"]["x"],
        center_y=sample["center"]["y"],
        x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
        footprint_px=p["footprint_px"],
        rotation_deg=p["rotation_deg"],
        scale=p["scale"],
        difficulty=difficulty,
        seed=int(seed),
        quad=sample["quad"],
    )
    return sample["reference_img"], sample["search_img"], gt


def to_rgb_variant(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn a grayscale capture into a 3-channel variant.

    Some detectors and viewers deliver colour, and a matcher that silently
    assumes single-channel input breaks on them. The channels are not identical
    copies: each gets its own small gain and offset and a little independent
    noise, which is what a real multi-channel readout looks like and what makes
    this a genuine test of the colour path rather than a formality.
    """
    if img.ndim == 3:
        return img
    base = img.astype(np.float32)
    channels = []
    for _ in range(3):
        gain = float(rng.uniform(0.94, 1.06))
        offset = float(rng.uniform(-6.0, 6.0))
        noise = rng.normal(0.0, 1.5, size=base.shape).astype(np.float32)
        channels.append(np.clip(base * gain + offset + noise, 0, 255))
    # OpenCV/PIL channel order differs; the harness only saves and re-reads
    # these, and `localize` converts back to grayscale, so stack plainly.
    return np.stack(channels, axis=-1).astype(np.uint8)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Inverse of `to_rgb_variant` for matcher input."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

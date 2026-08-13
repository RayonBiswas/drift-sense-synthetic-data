"""
Procedural DRAM-style layout generation and rasterization.

The layout is defined in *world units*, where one world unit equals one pixel of
the final 1000 x 1000 search image. It is rasterized onto a supersampled "fine
canvas" of size 1000 * S, which is then area-averaged down to 1000 x 1000 for
the search image. The reference image is cut from the same fine canvas at full
supersampled resolution, so it carries genuinely finer detail than the search
image rather than being an upscaled copy of it.

Structure drawn, from the bottom up:

  * peripheral / routing strips  -- flat mid-grey material with sparse
    orthogonal interconnect, filling the gaps between array blocks
  * array blocks ("mats")        -- the periodic cell array itself; each mat gets
    its own lattice phase and a small pitch multiplier, which is what makes one
    part of the search image distinguishable from another
  * word lines                   -- periodic horizontal lines
  * bit lines                    -- periodic vertical lines, ~1.5x the word-line
                                    pitch (the 6F^2 folded-bitline proportion)
  * storage-node contacts        -- dots on a checkerboard of the line
                                    intersections, one per two cells

Defects are injected during rasterization, not painted on afterwards, so they
appear consistently in both the reference and the search image -- they are
properties of the device, not of the capture.
"""

from __future__ import annotations

import cv2
import numpy as np

from .params import SEARCH_SIZE_PX
from .random import maybe_collapse_gap

# Grey levels of the material stack, chosen for high contrast between layers.
BACKGROUND = 32
WORD_LINE = 138
BIT_LINE = 168
CONTACT = 222
STRIP_BASE = 88
STRIP_LINE = 118

STRIP_ROUTING_PITCH = 26.0     # world units
STRIP_ROUTING_WIDTH = 2.2      # world units

WIDTH_JITTER_FRACTION = 0.09   # per-line CD variation


def _line_positions(extent: float, pitch: float, phase: float,
                    jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Line centre positions across `extent`, with a random lattice phase and a
    per-line random walk in spacing (line placement error)."""
    n = int(np.ceil(extent / max(pitch, 1e-6))) + 2
    idx = np.arange(-1, n, dtype=np.float64)
    pos = phase + idx * pitch
    if jitter > 0:
        pos = pos + rng.normal(0.0, jitter, size=pos.shape)
    return pos


def _line_mask(extent_px: int, positions_px: np.ndarray, width_px: float,
               rng: np.random.Generator, collapse_threshold_px: float = 0.0,
               defects: dict | None = None) -> tuple:
    """1D boolean mask of the lines, plus the per-line widths actually used.

    Returns (mask, widths). Widths carry per-instance CD variation, which is the
    line-width-variation defect mode in its always-on, low-amplitude form.

    Where the gap left between two neighbours falls below
    `collapse_threshold_px`, the pair may bridge: high-aspect-ratio lines topple
    into each other during processing. That is a property of the *device*, so a
    bridged pair shows up in the reference and the search image alike -- unlike
    noise, which is drawn separately per capture.
    """
    mask = np.zeros(extent_px, dtype=bool)
    if len(positions_px) == 0:
        return mask, np.zeros(0)
    widths = width_px * (1.0 + rng.normal(0.0, WIDTH_JITTER_FRACTION, size=len(positions_px)))
    widths = np.clip(widths, width_px * 0.45, width_px * 1.55)
    for i, (centre, w) in enumerate(zip(positions_px, widths)):
        lo = int(round(centre - w / 2.0))
        hi = int(round(centre + w / 2.0))
        if hi > 0 and lo < extent_px:
            mask[max(lo, 0):min(hi, extent_px)] = True

        if collapse_threshold_px <= 0 or i + 1 >= len(positions_px):
            continue
        next_centre, next_w = positions_px[i + 1], widths[i + 1]
        gap = (next_centre - next_w / 2.0) - (centre + w / 2.0)
        if maybe_collapse_gap(gap, collapse_threshold_px, rng):
            b_lo = int(round(centre + w / 2.0))
            b_hi = int(round(next_centre - next_w / 2.0))
            if b_hi > 0 and b_lo < extent_px and b_hi > b_lo:
                mask[max(b_lo, 0):min(b_hi, extent_px)] = True
                if defects is not None:
                    defects["collapsed_gap"] += 1
    return mask, widths


def _span_grid(size: float, block: float, strip: float) -> list:
    """Alternating [block, strip, block, strip, ...] spans covering `size`."""
    spans, pos, is_block = [], 0.0, True
    while pos < size:
        length = block if is_block else strip
        end = min(pos + length, size)
        spans.append((is_block, pos, end))
        pos = end
        is_block = not is_block
    return spans


def _strip_texture(n_fine: int, s: int, rng: np.random.Generator) -> np.ndarray:
    """Flat peripheral material with sparse orthogonal routing lines."""
    canvas = np.full((n_fine, n_fine), STRIP_BASE, dtype=np.uint8)
    half = STRIP_ROUTING_WIDTH * s / 2.0
    pitch = STRIP_ROUTING_PITCH * s
    for axis in (0, 1):
        start = rng.uniform(0, pitch)
        for centre in np.arange(start, n_fine, pitch):
            lo = max(int(round(centre - half)), 0)
            hi = min(int(round(centre + half)), n_fine)
            if hi <= lo:
                continue
            if axis == 0:
                canvas[lo:hi, :] = STRIP_LINE
            else:
                canvas[:, lo:hi] = STRIP_LINE
    return canvas


def _draw_mat(canvas: np.ndarray, y0: int, y1: int, x0: int, x1: int,
              params: dict, s: int, rng: np.random.Generator, defects: dict) -> None:
    """Rasterize one array block in place, into canvas[y0:y1, x0:x1]."""
    h, w = y1 - y0, x1 - x0
    if h <= 2 or w <= 2:
        return

    # Each mat gets its own lattice phase and a slight pitch trim. Without this
    # the whole canvas is one perfect lattice and no crop is distinguishable
    # from any other -- localization would be genuinely ill-posed.
    pitch_mult = float(rng.uniform(0.98, 1.02))
    word_pitch = params["word_line_pitch"] * pitch_mult * s
    bit_pitch = params["bit_line_pitch"] * pitch_mult * s
    phase_y = (params["phase_offset"][1] + rng.uniform(0, 1)) * word_pitch
    phase_x = (params["phase_offset"][0] + rng.uniform(0, 1)) * bit_pitch
    jitter = params["spacing_jitter"] * s

    word_pos = _line_positions(h, word_pitch, phase_y, jitter, rng)
    bit_pos = _line_positions(w, bit_pitch, phase_x, jitter, rng)

    word_width = params["line_width_frac"] * word_pitch
    bit_width = params["line_width_frac"] * bit_pitch

    collapse_px = params.get("collapse_threshold", 0.0) * s
    row_mask, _ = _line_mask(h, word_pos, word_width, rng, collapse_px, defects)
    col_mask, _ = _line_mask(w, bit_pos, bit_width, rng, collapse_px, defects)

    sub = canvas[y0:y1, x0:x1]
    sub[:] = BACKGROUND
    sub[row_mask, :] = WORD_LINE
    sub[:, col_mask] = np.maximum(sub[:, col_mask], BIT_LINE)

    density = params["defect_density"]

    # --- broken line: erase a segment of one line, leaving an open circuit --- #
    if density > 0:
        n_breaks = rng.poisson(density * 12.0)
        for _ in range(int(n_breaks)):
            if rng.random() < 0.5 and len(word_pos) > 0:
                centre = word_pos[rng.integers(0, len(word_pos))]
                lo = int(np.clip(centre - word_width, 0, h - 1))
                hi = int(np.clip(centre + word_width, 0, h))
                span = int(rng.uniform(0.05, 0.30) * w)
                cx = int(rng.uniform(0, max(w - span, 1)))
                sub[lo:hi, cx:cx + span] = BACKGROUND
            elif len(bit_pos) > 0:
                centre = bit_pos[rng.integers(0, len(bit_pos))]
                lo = int(np.clip(centre - bit_width, 0, w - 1))
                hi = int(np.clip(centre + bit_width, 0, w))
                span = int(rng.uniform(0.05, 0.30) * h)
                cy = int(rng.uniform(0, max(h - span, 1)))
                sub[cy:cy + span, lo:hi] = BACKGROUND
            defects["broken_line"] += 1

    # --- storage-node contacts on a checkerboard of the intersections ------- #
    radius_nom = max(params["contact_diameter_frac"] * word_pitch / 2.0, 1.0)
    parity = int(rng.integers(0, 2))
    for i, wy in enumerate(word_pos):
        if wy < -radius_nom or wy > h + radius_nom:
            continue
        for j, bx in enumerate(bit_pos):
            if bx < -radius_nom or bx > w + radius_nom:
                continue
            if (i + j) % 2 != parity:
                continue

            if density > 0 and rng.random() < density * 0.9:
                defects["missing_contact"] += 1        # unlanded / open contact
                continue

            cx, cy = bx, wy
            if density > 0 and rng.random() < density * 0.9:
                # misaligned contact: overlay-error style displacement
                cx += rng.normal(0, 0.22 * bit_pitch)
                cy += rng.normal(0, 0.22 * word_pitch)
                defects["displaced_contact"] += 1

            radius = radius_nom * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION))
            cv2.circle(sub, (int(round(cx)), int(round(cy))),
                       max(int(round(radius)), 1), CONTACT, -1)


def _apply_blob_defects(canvas: np.ndarray, params: dict, s: int,
                        rng: np.random.Generator, defects: dict) -> None:
    """Contamination particles and local intensity anomalies, drawn over the
    whole canvas rather than per mat."""
    density = params["defect_density"]
    if density <= 0:
        return
    n_fine = canvas.shape[0]

    n_contam = int(rng.poisson(density * 60.0))
    for _ in range(n_contam):
        cx, cy = rng.integers(0, n_fine, size=2)
        rad = int(rng.uniform(1.5, 7.0) * s)
        axes = (max(rad, 1), max(int(rad * rng.uniform(0.5, 1.5)), 1))
        angle = float(rng.uniform(0, 180))
        val = int(np.clip(rng.uniform(200, 255), 0, 255))
        cv2.ellipse(canvas, (int(cx), int(cy)), axes, angle, 0, 360, val, -1)
        defects["contamination"] += 1

    n_dim = int(rng.poisson(density * 40.0))
    for _ in range(n_dim):
        cx, cy = rng.integers(0, n_fine, size=2)
        rad = int(rng.uniform(3.0, 12.0) * s)
        y0, y1 = max(int(cy - rad), 0), min(int(cy + rad), n_fine)
        x0, x1 = max(int(cx - rad), 0), min(int(cx + rad), n_fine)
        if y1 <= y0 or x1 <= x0:
            continue
        patch = canvas[y0:y1, x0:x1].astype(np.float32)
        gain = float(rng.uniform(0.55, 0.85)) if rng.random() < 0.7 else float(rng.uniform(1.15, 1.45))
        canvas[y0:y1, x0:x1] = np.clip(patch * gain, 0, 255).astype(np.uint8)
        defects["intensity_anomaly"] += 1


def render_fine_canvas(params: dict, supersample: int, seed: int) -> tuple:
    """Rasterize the full layout at 1000 * supersample resolution.

    Returns (canvas, defect_counts). The canvas is uint8 and is the *specimen*:
    no imaging effects have been applied to it yet.
    """
    s = int(supersample)
    n_fine = SEARCH_SIZE_PX * s
    rng = np.random.default_rng(int(seed))

    defects = {
        "broken_line": 0,
        "missing_contact": 0,
        "displaced_contact": 0,
        "contamination": 0,
        "intensity_anomaly": 0,
        "collapsed_gap": 0,
    }

    canvas = _strip_texture(n_fine, s, rng)

    row_spans = _span_grid(SEARCH_SIZE_PX, params["block_size"], params["strip_width"])
    col_spans = _span_grid(SEARCH_SIZE_PX, params["block_size"], params["strip_width"])

    for row_is_block, wy0, wy1 in row_spans:
        if not row_is_block:
            continue
        for col_is_block, wx0, wx1 in col_spans:
            if not col_is_block:
                continue
            _draw_mat(
                canvas,
                int(round(wy0 * s)), int(round(wy1 * s)),
                int(round(wx0 * s)), int(round(wx1 * s)),
                params, s, rng, defects,
            )

    _apply_blob_defects(canvas, params, s, rng, defects)
    return canvas, defects
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
# The two architectures use near-identical levels -- both are "dark substrate,
# two brighter line layers, brightest contacts" -- so they are kept as one table
# keyed by architecture rather than as two divergent sets of constants.
BACKGROUND = 32
WORD_LINE = 138
BIT_LINE = 168
CONTACT = 222
STRIP_BASE = 88
STRIP_LINE = 118

# FinFET stack: substrate / fin / gate / contact.
FINFET_BACKGROUND = 40
FIN = 150
GATE = 170
FINFET_CONTACT = 225

LEVELS = {
    "dram": {
        "background": BACKGROUND,
        "primary": WORD_LINE,      # word lines, horizontal
        "secondary": BIT_LINE,     # bit lines, vertical
        "contact": CONTACT,
    },
    "finfet": {
        "background": FINFET_BACKGROUND,
        "primary": GATE,           # gate stripes, horizontal
        "secondary": FIN,          # fins, vertical
        "contact": FINFET_CONTACT,
    },
}

STRIP_ROUTING_PITCH = 26.0     # world units
STRIP_ROUTING_WIDTH = 2.2      # world units

WIDTH_JITTER_FRACTION = 0.09   # per-line CD variation
MAT_PITCH_TRIM = 0.02          # per-mat pitch deviation from nominal, +/- fraction


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


def _contact_sites(arch: str, row_pos: np.ndarray, col_pos: np.ndarray,
                   h: int, w: int, margin: float, parity: int) -> list:
    """Where the contacts go, as (x, y) centres in mat-local coordinates.

    This is the one place the two architectures genuinely diverge:

      dram    a storage-node contact lands on the *intersection* of a word line
              and a bit line -- one per two cells, hence the checkerboard.
      finfet  a source/drain contact lands on a fin, in the diffusion gap
              *between* two consecutive gate stripes -- never on a gate.

    Emitted in (row index, column index) order for DRAM so the per-contact
    random draws downstream keep the sequence they had before FinFET existed.
    """
    sites = []
    if arch == "dram":
        for i, y in enumerate(row_pos):
            if y < -margin or y > h + margin:
                continue
            for j, x in enumerate(col_pos):
                if x < -margin or x > w + margin:
                    continue
                if (i + j) % 2 != parity:
                    continue
                sites.append((x, y))
        return sites

    # finfet: iterate fins (columns), then the gaps between gate stripes (rows)
    for i, x in enumerate(col_pos):
        if x < -margin or x > w + margin:
            continue
        for j in range(len(row_pos) - 1):
            if (i + j) % 2 != parity:
                continue
            y = (row_pos[j] + row_pos[j + 1]) / 2.0
            if y < -margin or y > h + margin:
                continue
            sites.append((x, y))
    return sites


def _draw_mat(canvas: np.ndarray, y0: int, y1: int, x0: int, x1: int,
              params: dict, s: int, rng: np.random.Generator, defects: dict) -> None:
    """Rasterize one array block in place, into canvas[y0:y1, x0:x1].

    Both architectures are the same drawing problem -- a set of horizontal lines,
    a set of vertical lines, and contacts on a checkerboard -- so they share one
    renderer. Only three things vary: which pitch runs which way, the grey
    levels, and where the contacts sit relative to the lines.

        dram    horizontal word lines at P01, vertical bit lines at P01 x P02
        finfet  vertical fins at P01, horizontal gate stripes at P01 x P02

    The draw sequence is identical in both cases, so a seed produces the same
    lattice phases, jitter and defect pattern regardless of architecture.
    """
    h, w = y1 - y0, x1 - x0
    if h <= 2 or w <= 2:
        return

    arch = params.get("architecture", "dram")
    levels = LEVELS[arch]

    # Each mat gets its own lattice phase and a slight pitch trim. Without this
    # the whole canvas is one perfect lattice and no crop is distinguishable
    # from any other -- localization would be genuinely ill-posed.
    pitch_mult = float(rng.uniform(1.0 - MAT_PITCH_TRIM, 1.0 + MAT_PITCH_TRIM))
    if arch == "dram":
        # word lines run horizontally at the primary pitch
        row_pitch = params["word_line_pitch"] * pitch_mult * s
        col_pitch = params["bit_line_pitch"] * pitch_mult * s
    else:
        # fins run vertically at the primary pitch; gates are the wider CPP
        row_pitch = params["bit_line_pitch"] * pitch_mult * s
        col_pitch = params["word_line_pitch"] * pitch_mult * s
    phase_y = (params["phase_offset"][1] + rng.uniform(0, 1)) * row_pitch
    phase_x = (params["phase_offset"][0] + rng.uniform(0, 1)) * col_pitch
    jitter = params["spacing_jitter"] * s

    row_pos = _line_positions(h, row_pitch, phase_y, jitter, rng)
    col_pos = _line_positions(w, col_pitch, phase_x, jitter, rng)

    row_width = params["line_width_frac"] * row_pitch
    col_width = params["line_width_frac"] * col_pitch

    collapse_px = params.get("collapse_threshold", 0.0) * s
    row_mask, _ = _line_mask(h, row_pos, row_width, rng, collapse_px, defects)
    col_mask, _ = _line_mask(w, col_pos, col_width, rng, collapse_px, defects)

    sub = canvas[y0:y1, x0:x1]
    sub[:] = levels["background"]
    sub[row_mask, :] = levels["primary"]
    sub[:, col_mask] = np.maximum(sub[:, col_mask], levels["secondary"])

    density = params["defect_density"]

    # --- broken line: erase a segment of one line, leaving an open circuit --- #
    if density > 0:
        n_breaks = rng.poisson(density * 12.0)
        for _ in range(int(n_breaks)):
            if rng.random() < 0.5 and len(row_pos) > 0:
                centre = row_pos[rng.integers(0, len(row_pos))]
                lo = int(np.clip(centre - row_width, 0, h - 1))
                hi = int(np.clip(centre + row_width, 0, h))
                span = int(rng.uniform(0.05, 0.30) * w)
                cx = int(rng.uniform(0, max(w - span, 1)))
                sub[lo:hi, cx:cx + span] = levels["background"]
            elif len(col_pos) > 0:
                centre = col_pos[rng.integers(0, len(col_pos))]
                lo = int(np.clip(centre - col_width, 0, w - 1))
                hi = int(np.clip(centre + col_width, 0, w))
                span = int(rng.uniform(0.05, 0.30) * h)
                cy = int(rng.uniform(0, max(h - span, 1)))
                sub[cy:cy + span, lo:hi] = levels["background"]
            defects["broken_line"] += 1

    # --- contacts on a checkerboard ---------------------------------------- #
    # Sized off the primary pitch in both cases: the storage-node landing pad
    # scales with the word-line pitch, the source/drain contact with the fin
    # pitch. That is `word_line_pitch` either way, which is row_pitch for DRAM
    # and col_pitch for FinFET.
    primary_pitch = row_pitch if arch == "dram" else col_pitch
    radius_nom = max(params["contact_diameter_frac"] * primary_pitch / 2.0, 1.0)
    parity = int(rng.integers(0, 2))

    for cx, cy in _contact_sites(arch, row_pos, col_pos, h, w, radius_nom, parity):
        if density > 0 and rng.random() < density * 0.9:
            defects["missing_contact"] += 1            # unlanded / open contact
            continue

        if density > 0 and rng.random() < density * 0.9:
            # misaligned contact: overlay-error style displacement
            cx += rng.normal(0, 0.22 * col_pitch)
            cy += rng.normal(0, 0.22 * row_pitch)
            defects["displaced_contact"] += 1

        radius = radius_nom * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION))
        r = max(int(round(radius)), 1)
        ix, iy = int(round(cx)), int(round(cy))
        if arch == "dram":
            # round storage-node pad
            cv2.circle(sub, (ix, iy), r, levels["contact"], -1)
        else:
            # square source/drain contact bar
            cv2.rectangle(sub, (ix - r, iy - r), (ix + r, iy + r),
                          levels["contact"], -1)


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
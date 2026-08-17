"""
Pattern-collapse / bridging.

This models a real *structural* defect (high-aspect-ratio lines toppling and
sticking to their neighbor, e.g. resist/fin collapse from capillary forces)
-- it is applied once to the fine canvas and therefore shows up consistently
in both the reference crop and the derived search image. It is distinct from
SEM *imaging* artifacts (blur/noise/drift), which live in sem_imaging.py and
are applied per-image since reference and search are captured differently.

Default collapse_threshold_nm=10 is tied to the search image's 10 nm/px
resolution: a 10 nm gap is exactly 1 px there, i.e. right at the edge of what
that image can physically resolve as two separate structures.

The implementation is the canonical engine's -- the two copies were verified to
produce identical decisions over 2000 draws from the same seed before being
collapsed into one. Only the argument *names* differed (nm here, world units
there); the rule is the same, so it is re-exported rather than reimplemented.
"""

from dram_dataset.dram_synth.random import maybe_collapse_gap  # noqa: F401

__all__ = ["maybe_collapse_gap"]

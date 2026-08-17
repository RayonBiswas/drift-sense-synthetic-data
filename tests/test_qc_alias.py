"""
Controls on the QC lattice-alias test.

Check 10 asks whether a correlation probe that missed the annotation landed on
a neighbouring repeat of the same pattern. The budget it allows has to sit in a
narrow band:

  too tight  legitimate repeats get reported as broken ground truth, because
             the lattice in the image is not the ideal lattice the annotation
             records -- vibration alone displaces two raster bands by up to
             twice its amplitude
  too loose  an offset sitting exactly *between* two repeats gets certified as
             a repeat, and the check stops meaning anything

The second failure is the easy one to introduce while fixing the first, so it
is tested explicitly: a half-pitch offset is maximally not-on-a-repeat and must
never be accepted, at any pitch this generator produces.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "dram_dataset"))

from dram_synth.qc import (  # noqa: E402
    GT_ALIAS_MAX_STEPS,
    GT_ALIAS_PITCH_FRACTION,
    _alias_residual_budget,
    _is_lattice_alias,
)

# Spans the generator's range of pitches and scan-motion severities.
PITCHES = [(12.6, 7.8), (15.3, 10.4), (19.0, 13.5), (11.0, 6.5)]
VIBRATIONS = [0.0, 0.8, 1.6, 2.4]


def record(bit_pitch, word_pitch, vibration=1.6, jitter=0.15,
           thermal=3.0, shear=5.0):
    return {
        "architecture": {
            "bit_line_pitch": bit_pitch,
            "P01_word_line_pitch": word_pitch,
            "P06_spacing_jitter": jitter,
        },
        "search_imaging": {
            "vibration_amp_px": vibration,
            "vibration_jitter_px": 0.2,
            "thermal_drift_px": thermal,
            "raster_shear_px": shear,
        },
    }


ALL_RECORDS = [record(bp, wp, vibration=v)
               for bp, wp in PITCHES for v in VIBRATIONS]


@pytest.mark.parametrize("rec", ALL_RECORDS)
@pytest.mark.parametrize("steps", range(1, GT_ALIAS_MAX_STEPS + 1))
def test_exact_repeats_are_accepted(rec, steps):
    """A peak sitting exactly on the n-th repeat is a repeat."""
    word = rec["architecture"]["P01_word_line_pitch"]
    assert _is_lattice_alias({"dx": 0.0, "dy": steps * word}, rec)


@pytest.mark.parametrize("rec", ALL_RECORDS)
def test_half_pitch_offsets_are_never_accepted(rec):
    """Maximally off-repeat, on either axis or both. This is the control that
    catches a budget grown wider than the lattice can resolve."""
    bit = rec["architecture"]["bit_line_pitch"]
    word = rec["architecture"]["P01_word_line_pitch"]
    for dx, dy in ((0.0, 3 * word + word / 2),
                   (2 * bit + bit / 2, 0.0),
                   (2 * bit + bit / 2, 3 * word + word / 2)):
        assert not _is_lattice_alias({"dx": dx, "dy": dy}, rec), (
            f"half-pitch offset ({dx:.1f}, {dy:.1f}) accepted as a repeat")


@pytest.mark.parametrize("rec", ALL_RECORDS)
def test_budget_never_exceeds_what_the_pitch_can_resolve(rec):
    """The ceiling is what keeps the previous test passing."""
    smallest = min(rec["architecture"]["bit_line_pitch"],
                   rec["architecture"]["P01_word_line_pitch"])
    for nx in range(GT_ALIAS_MAX_STEPS + 1):
        for ny in range(GT_ALIAS_MAX_STEPS + 1):
            budget = _alias_residual_budget(rec, nx, ny)
            assert budget <= GT_ALIAS_PITCH_FRACTION * smallest + 1e-9
            assert budget < smallest / 2.0, "budget must stay under half a pitch"


def test_budget_grows_with_scan_motion():
    """A sample with more stage vibration earns a wider budget, up to the cap --
    this is the whole reason the budget is derived rather than flat."""
    calm = _alias_residual_budget(record(19.0, 13.5, vibration=0.0), 0, 2)
    shaky = _alias_residual_budget(record(19.0, 13.5, vibration=2.4), 0, 2)
    assert shaky > calm


def test_offsets_beyond_max_steps_are_rejected():
    rec = record(12.6, 7.8)
    word = rec["architecture"]["P01_word_line_pitch"]
    assert not _is_lattice_alias(
        {"dx": 0.0, "dy": (GT_ALIAS_MAX_STEPS + 1) * word}, rec)


def test_degenerate_pitch_is_rejected_not_crashed():
    rec = record(0.0, 0.0)
    assert not _is_lattice_alias({"dx": 0.0, "dy": 0.0}, rec)

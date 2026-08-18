"""
Step 1 gate: the spec's tie-break rule.

    "If more than one matching region is found, return the one closest to the
     center of the Search Image."

Two levels of test. The first drives `_find_candidates` on a hand-built
correlation surface, so peak extraction is checked without any imaging in the
way. The second runs the whole matcher on a search image that genuinely
contains three identical copies of the reference, which is the situation the
rule exists for.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "dram_dataset"))

from localize import (  # noqa: E402
    CANDIDATE_MIN_DISTANCE,
    _find_candidates,
    localize,
)

SEARCH_SIZE = 1000
PATCH = 100
# Patch centres. (520, 520) is nearest the 1000x1000 image centre at (500, 500).
PLACEMENTS = {"corner": (200, 200), "near_center": (520, 520), "right": (800, 300)}


def build_scene(placements, seed=7):
    """A noise background with `placements` copies of one noise patch in it.

    Noise rather than a lattice: the point here is to test the tie-break, so the
    peaks must be unambiguous and exactly equal. Periodic-array ambiguity is
    measured on the real dataset, not simulated here.
    """
    rng = np.random.default_rng(seed)
    search = rng.integers(0, 256, (SEARCH_SIZE, SEARCH_SIZE), dtype=np.uint8)
    patch = rng.integers(0, 256, (PATCH, PATCH), dtype=np.uint8)
    for cx, cy in placements:
        x0, y0 = cx - PATCH // 2, cy - PATCH // 2
        search[y0:y0 + PATCH, x0:x0 + PATCH] = patch
    return patch, search


def test_find_candidates_extracts_every_peak_centre():
    """Three planted peaks on a flat surface come back, strongest first."""
    surface = np.zeros((400, 400), dtype=np.float32)
    surface[100, 100] = 0.90
    surface[300, 300] = 1.00
    surface[100, 300] = 0.96
    # 0.80 is below 1.00 * CANDIDATE_RATIO (0.95), so it must be excluded.
    surface[200, 50] = 0.80

    k = 20
    found = _find_candidates(surface, best=1.00, k=k)

    assert len(found) == 2, "only peaks within 5% of the best score qualify"
    assert [round(c.score, 2) for c in found] == [1.00, 0.96], "strongest first"
    assert (found[0].center_x, found[0].center_y) == (300 + k / 2, 300 + k / 2)


def test_find_candidates_suppresses_a_plateau():
    """A block of equal values is one peak, not one peak per pixel."""
    surface = np.zeros((200, 200), dtype=np.float32)
    surface[100:104, 100:104] = 1.0          # 16 adjacent equal pixels
    found = _find_candidates(surface, best=1.0, k=10)
    assert len(found) == 1, f"plateau reported as {len(found)} peaks"


def test_find_candidates_keeps_peaks_beyond_min_distance():
    surface = np.zeros((200, 200), dtype=np.float32)
    surface[100, 100] = 1.0
    surface[100, 100 + CANDIDATE_MIN_DISTANCE + 2] = 1.0
    assert len(_find_candidates(surface, best=1.0, k=10)) == 2


def test_center_tiebreak_returns_the_center_most_match():
    """The rule that the spec actually asks for."""
    patch, search = build_scene(PLACEMENTS.values())
    result = localize(patch, search, tie_break="center")

    assert result.n_candidates >= 3, (
        f"scene has 3 identical copies but only {result.n_candidates} candidates "
        f"were found")
    want_x, want_y = PLACEMENTS["near_center"]
    assert abs(result.center_x - want_x) <= 3
    assert abs(result.center_y - want_y) <= 3


def test_score_tiebreak_still_returns_the_strongest_peak():
    """The old behaviour is preserved and selectable, for comparison."""
    patch, search = build_scene(PLACEMENTS.values())
    result = localize(patch, search, tie_break="score")

    assert result.candidates, "candidates should always be populated"
    best = max(result.candidates, key=lambda c: c.score)
    assert result.center_x == best.center_x
    assert result.center_y == best.center_y


def test_the_two_policies_disagree_on_this_scene():
    """Guards the test itself: if both policies picked the same peak, the
    center test above would pass without exercising the rule at all."""
    patch, search = build_scene(PLACEMENTS.values())
    by_center = localize(patch, search, tie_break="center")
    by_score = localize(patch, search, tie_break="score")
    assert (by_center.center_x, by_center.center_y) != (by_score.center_x, by_score.center_y)
    assert by_center.tie_break_applied == "center"
    assert by_score.tie_break_applied == "score"


def test_single_match_is_unaffected_by_the_rule():
    """With one matching region there is nothing to tie-break."""
    patch, search = build_scene([PLACEMENTS["corner"]])
    by_center = localize(patch, search, tie_break="center")
    by_score = localize(patch, search, tie_break="score")

    assert by_center.n_candidates == 1
    assert by_center.tie_break_applied == "score", "rule must not fire on one match"
    assert (by_center.center_x, by_center.center_y) == (by_score.center_x, by_score.center_y)
    assert abs(by_center.center_x - PLACEMENTS["corner"][0]) <= 3
    assert abs(by_center.center_y - PLACEMENTS["corner"][1]) <= 3


def test_candidates_length_matches_the_reported_count():
    patch, search = build_scene(PLACEMENTS.values())
    result = localize(patch, search)
    assert len(result.candidates) == result.n_candidates


def test_center_is_the_default_policy():
    """The spec rule should be what you get without asking for it."""
    patch, search = build_scene(PLACEMENTS.values())
    assert (localize(patch, search).center_x
            == localize(patch, search, tie_break="center").center_x)


def test_unknown_tie_break_is_rejected():
    patch, search = build_scene([PLACEMENTS["corner"]])
    with pytest.raises(ValueError, match="tie_break"):
        localize(patch, search, tie_break="closest")

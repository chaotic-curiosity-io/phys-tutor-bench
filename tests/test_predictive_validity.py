"""Tests for the process->transfer predictive-validity probe.

The statistics (Pearson/Spearman) are reused from already-tested helpers; what is
novel here and worth pinning down is (a) the process composite must EXCLUDE the
transfer-success outcome and renormalize the rubric weights over the remaining five
dimensions, and (b) the cross-judge alignment must inner-join on conversation_id and
take the predictor from one judge and the outcome from the other.
"""

from __future__ import annotations

import math

from src.scenarios.schema import ConversationScore, DimensionScore
from src.scoring.rubric import DEFAULT_WEIGHTS
from src.validation.predictive_validity import (
    PROCESS_DIMENSIONS,
    OUTCOME_DIMENSION,
    dimension_map,
    process_composite,
    align_predictor_outcome,
    within_judge,
    range_restriction_correct,
)


def _score(cid: str, judge: str, dims: dict[str, int]) -> ConversationScore:
    return ConversationScore(
        conversation_id=cid,
        scenario_id="mech-test-001",
        model_under_test="some-tutor",
        judge_model=judge,
        scores=[DimensionScore(dimension=d, score=s, justification="x") for d, s in dims.items()],
        composite_score=0.0,
        timestamp="2026-01-01T00:00:00+00:00",
    )


ALL_FOUR = {
    "misconception_diagnosis": 4,
    "scaffolding_strategy": 4,
    "answer_disclosure_restraint": 4,
    "conceptual_model_building": 4,
    "transfer_success": 4,
    "pedagogical_harm_avoidance": 4,
}


def test_outcome_is_transfer_and_excluded_from_process():
    assert OUTCOME_DIMENSION == "transfer_success"
    assert OUTCOME_DIMENSION not in PROCESS_DIMENSIONS
    assert len(PROCESS_DIMENSIONS) == 5


def test_process_composite_ignores_transfer_success():
    # A perfect process record with a FAILED transfer must still score 4.0 on process.
    dims = dict(ALL_FOUR, transfer_success=0)
    assert math.isclose(process_composite(dimension_map(_score("a", "j", dims))), 4.0, rel_tol=1e-9)


def test_process_composite_renormalizes_weights_over_five_dims():
    # Only MD=4, everything else 0. Weighted-over-5 = (4 * w_MD) / sum(non-TS weights).
    dims = {d: 0 for d in ALL_FOUR}
    dims["misconception_diagnosis"] = 4
    non_ts_weight = sum(w for k, w in DEFAULT_WEIGHTS.items() if k != "transfer_success")
    expected = (4 * DEFAULT_WEIGHTS["misconception_diagnosis"]) / non_ts_weight
    assert math.isclose(process_composite(dimension_map(_score("a", "j", dims))), expected, rel_tol=1e-9)


def test_align_inner_joins_and_crosses_judges():
    # Predictor judge A: process varies; outcome judge B: TS varies. Only b,c overlap.
    a = [
        _score("a", "A", dict(ALL_FOUR)),
        _score("b", "A", dict(ALL_FOUR, misconception_diagnosis=0, scaffolding_strategy=0,
                              answer_disclosure_restraint=0, conceptual_model_building=0,
                              pedagogical_harm_avoidance=0)),  # process = 0
        _score("c", "A", dict(ALL_FOUR)),                      # process = 4
    ]
    b = [
        _score("b", "B", dict(ALL_FOUR, transfer_success=1)),
        _score("c", "B", dict(ALL_FOUR, transfer_success=3)),
        _score("d", "B", dict(ALL_FOUR, transfer_success=4)),
    ]
    predictors, outcomes, cids = align_predictor_outcome(a, b)
    assert cids == ["b", "c"]                 # inner join, sorted; "a" and "d" dropped
    assert math.isclose(predictors[0], 0.0)   # b: predictor from judge A's process
    assert math.isclose(predictors[1], 4.0)   # c: predictor from judge A's process
    assert outcomes == [1, 3]                  # b,c: outcome = judge B's transfer_success


def test_within_judge_does_not_dedupe_shared_conversation_ids():
    # The same conversation scored by two judges must count as TWO points, not collapse to one.
    shared = "same-cid"
    scores = [
        _score(shared, "judgeA", dict(ALL_FOUR, transfer_success=4)),
        _score(shared, "judgeB", dict(ALL_FOUR, transfer_success=0)),
    ]
    assert within_judge(scores)["n"] == 2


def test_range_restriction_correction_recovers_known_value():
    # Thorndike Case II: r=0.5, SD ratio 2 -> 2*0.5 / sqrt(1 - 0.25 + 0.25*4) = 1/sqrt(1.75).
    got = range_restriction_correct(0.5, restricted_sd=1.0, unrestricted_sd=2.0)
    assert math.isclose(got, 1.0 / math.sqrt(1.75), rel_tol=1e-9)
    # Correcting for a WIDER unrestricted range increases the estimate.
    assert got > 0.5
    assert range_restriction_correct(None, 1.0, 2.0) is None

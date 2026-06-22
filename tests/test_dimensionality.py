"""Tests for the dimension-redundancy / construct-validity probe.

The novel, get-it-wrong-and-the-finding-is-wrong logic: Cronbach's alpha, the eigenvalue
summary of a correlation matrix, and the cross-judge multitrait-multimethod (MTMM) matrix
whose diagonal is each dimension's cross-judge convergent reliability.
"""

from __future__ import annotations

import math

import numpy as np

from src.scenarios.schema import ConversationScore, DimensionScore
from src.validation.dimensionality import (
    DIM_IDS,
    cronbach_alpha,
    eigen_summary,
    cross_judge_mtmm,
    mtmm_summary,
    redundancy_pairs,
    mean_offdiag,
)


def _cs(cid, judge, dim_scores):
    return ConversationScore(
        conversation_id=cid, scenario_id="s", model_under_test="t", judge_model=judge,
        scores=[DimensionScore(dimension=d, score=dim_scores[d], justification="x") for d in DIM_IDS],
        composite_score=0.0, timestamp="2026-01-01T00:00:00+00:00",
    )


def _uniform(val):
    return {d: val for d in DIM_IDS}


def test_cronbach_alpha_is_one_for_identical_items():
    # Every dimension carries the same varying value -> six identical items -> alpha = 1.0.
    scores = [_cs(f"c{v}", "j", _uniform(v)) for v in (0, 1, 2, 3, 4)]
    assert math.isclose(cronbach_alpha(scores), 1.0, rel_tol=1e-9)


def test_cronbach_alpha_none_when_no_variance():
    scores = [_cs(f"c{i}", "j", _uniform(2)) for i in range(4)]   # all constant
    assert cronbach_alpha(scores) is None


def test_eigen_summary_identity_and_rank_one():
    ident = eigen_summary(np.eye(6))
    assert math.isclose(ident["pc1_frac"], 1 / 6, rel_tol=1e-9)
    assert ident["kaiser_factors"] == 0           # no eigenvalue strictly > 1
    rank1 = eigen_summary(np.ones((6, 6)))
    assert math.isclose(rank1["pc1_frac"], 1.0, rel_tol=1e-9)   # one factor explains everything


def test_cross_judge_mtmm_diagonal_is_same_dimension_reliability():
    # Judges agree exactly; only MD (index 0) varies across conversations.
    rows_a, rows_b = [], []
    for i, v in enumerate((0, 1, 2, 3)):
        d = _uniform(2); d[DIM_IDS[0]] = v
        rows_a.append(_cs(f"c{i}", "A", d))
        rows_b.append(_cs(f"c{i}", "B", d))
    M, n = cross_judge_mtmm(rows_a, rows_b)
    assert M.shape == (6, 6) and n == 4
    assert math.isclose(M[0][0], 1.0, rel_tol=1e-9)   # A's MD vs B's MD, identical & varying


def test_mtmm_summary_reports_convergent_and_heterotrait():
    M = np.full((6, 6), 0.2)
    np.fill_diagonal(M, 0.9)
    s = mtmm_summary(M)
    assert math.isclose(s["mean_convergent"], 0.9, rel_tol=1e-9)
    assert math.isclose(s["mean_heterotrait"], 0.2, rel_tol=1e-9)
    assert s["campbell_fiske_violations"] == 0       # every diagonal exceeds its off-diagonals


def test_redundancy_pairs_flags_high_correlation():
    m = np.eye(6)
    m[1][2] = m[2][1] = 0.88
    pairs = redundancy_pairs(m, threshold=0.8)
    assert len(pairs) == 1 and math.isclose(pairs[0][2], 0.88, rel_tol=1e-9)
    assert mean_offdiag(np.eye(6)) == 0.0

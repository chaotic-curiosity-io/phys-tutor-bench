"""Tests for the judge self-preference (own-family bias) analysis.

The load-bearing, get-it-wrong-and-the-finding-is-wrong logic: (a) mapping a model name
to a provider family, (b) aligning the two judges' scores on the same conversation, and
(c) the difference-in-differences self-preference estimate and its permutation test. The
DiD is built on the paired within-conversation delta (judge A - judge B), so conversation
difficulty cancels; what remains is whether that delta is larger for judge A's own family.
"""

from __future__ import annotations

import math

from src.scenarios.schema import ConversationScore, DimensionScore
from src.validation.judge_bias import (
    model_family,
    tier_of,
    align_judges,
    paired_deltas,
    difference_in_differences,
    permutation_test,
    ceiling_rates,
    tier_controlled_family_effect,
    stratified_permutation_test,
)

DIMS = ["misconception_diagnosis", "scaffolding_strategy", "answer_disclosure_restraint",
        "conceptual_model_building", "transfer_success", "pedagogical_harm_avoidance"]


def _cs(cid, tutor, judge, comp, dim_score=3):
    return ConversationScore(
        conversation_id=cid, scenario_id="s", model_under_test=tutor, judge_model=judge,
        scores=[DimensionScore(dimension=d, score=dim_score, justification="x") for d in DIMS],
        composite_score=comp, timestamp="2026-01-01T00:00:00+00:00",
    )


def test_model_family_maps_provider_prefixes():
    assert model_family("claude-opus-4-8") == "anthropic"
    assert model_family("gpt-4o") == "openai"
    assert model_family("gpt-5.5") == "openai"
    assert model_family("o3-mini") == "openai"
    assert model_family("llama3.1:8b") == "other"


def test_align_joins_on_conversation_and_keeps_tutor():
    a = [_cs("x", "claude-opus-4-8", "claude-opus-4-8", 4.0), _cs("y", "gpt-4o", "claude-opus-4-8", 3.0)]
    b = [_cs("x", "claude-opus-4-8", "gpt-5.5", 3.5), _cs("z", "gpt-4o", "gpt-5.5", 2.0)]
    rows = align_judges(a, b)
    assert [r["conversation_id"] for r in rows] == ["x"]   # inner join; y, z dropped
    assert rows[0]["tutor"] == "claude-opus-4-8"
    assert rows[0]["family"] == "anthropic"
    assert rows[0]["a_comp"] == 4.0 and rows[0]["b_comp"] == 3.5


def test_did_is_positive_when_judge_a_inflates_own_family():
    # Judge A (anthropic) scores anthropic tutors +0.5 over judge B, openai tutors +0.0.
    rows = [
        _cs("a1", "claude-opus-4-8", "A", 4.0), _cs("a2", "claude-sonnet-4-6", "A", 3.5),
        _cs("o1", "gpt-4o", "A", 3.0), _cs("o2", "gpt-4o", "A", 2.0),
    ]
    rowsB = [
        _cs("a1", "claude-opus-4-8", "B", 3.5), _cs("a2", "claude-sonnet-4-6", "B", 3.0),
        _cs("o1", "gpt-4o", "B", 3.0), _cs("o2", "gpt-4o", "B", 2.0),
    ]
    aligned = align_judges(rows, rowsB)
    did = difference_in_differences(aligned, family_a="anthropic", family_b="openai")
    assert abs(did - 0.5) < 1e-9          # +0.5 own-family inflation, 0 for outgroup

    # Control: judge A is uniformly +0.3 over B -> no family-specific effect -> DiD ~ 0.
    rowsB2 = [_cs(r["conversation_id"], r["tutor"], "B", r["a_comp"] - 0.3) for r in aligned]
    did0 = difference_in_differences(align_judges(rows, rowsB2), family_a="anthropic", family_b="openai")
    assert abs(did0) < 1e-9


def test_permutation_test_is_deterministic_and_flags_strong_effect():
    rowsA = [_cs(f"a{i}", "claude-opus-4-8", "A", 4.0) for i in range(8)] + \
            [_cs(f"o{i}", "gpt-4o", "A", 3.0) for i in range(8)]
    rowsB = [_cs(f"a{i}", "claude-opus-4-8", "B", 3.0) for i in range(8)] + \
            [_cs(f"o{i}", "gpt-4o", "B", 3.0) for i in range(8)]
    aligned = align_judges(rowsA, rowsB)
    r1 = permutation_test(aligned, "anthropic", "openai", n_perm=2000, seed=0)
    r2 = permutation_test(aligned, "anthropic", "openai", n_perm=2000, seed=0)
    assert r1["observed"] == r2["observed"] and r1["p_value"] == r2["p_value"]  # deterministic
    assert abs(r1["observed"] - 1.0) < 1e-9    # +1.0 inflation on own family only
    assert r1["p_value"] < 0.05                # clearly non-null


def test_ceiling_rate_counts_max_scores_per_judge_family():
    a = [_cs("x", "claude-opus-4-8", "A", 4.0, dim_score=4)]   # all six dims == 4
    b = [_cs("x", "claude-opus-4-8", "B", 3.0, dim_score=2)]   # none == 4
    rates = ceiling_rates(align_judges(a, b))
    assert rates["a"]["anthropic"] == 1.0     # judge A: 6/6 at ceiling
    assert rates["b"]["anthropic"] == 0.0     # judge B: 0/6


def test_tier_of_maps_models():
    assert tier_of("claude-opus-4-8") == "strong"
    assert tier_of("gpt-5.5") == "strong"
    assert tier_of("gpt-4o") == "weak"
    assert tier_of("claude-haiku-4-5-weak") == "weak"
    assert tier_of("mystery-model") == "unknown"


def _row(family, tier, delta, comp_b=3.0):
    # a row as produced by align_judges; these analyses only read family/tier/a_comp/b_comp.
    return {"family": family, "tier": tier, "a_comp": comp_b + delta, "b_comp": comp_b,
            "tutor": f"{family}-{tier}", "conversation_id": "x", "a_dims": {}, "b_dims": {}}


def test_tier_controlled_effect_averages_within_tier_contrasts():
    rows = (
        [_row("anthropic", "strong", 0.5)] * 2 + [_row("openai", "strong", 0.1)] * 2 +  # strong contrast 0.4
        [_row("anthropic", "weak", 0.3)] * 2 + [_row("openai", "weak", 0.1)] * 2        # weak contrast 0.2
    )
    res = tier_controlled_family_effect(rows, "anthropic", "openai")
    assert math.isclose(res["per_tier_contrast"]["strong"], 0.4, rel_tol=1e-9)
    assert math.isclose(res["per_tier_contrast"]["weak"], 0.2, rel_tol=1e-9)
    assert math.isclose(res["tier_controlled_effect"], 0.3, rel_tol=1e-9)


def test_tier_control_collapses_a_pure_quality_confound():
    # Delta is driven by TIER (strong=0.4, weak=0.0), and family is confounded with tier
    # (anthropic mostly strong, openai mostly weak). Naive DiD looks positive; controlled = 0.
    rows = (
        [_row("anthropic", "strong", 0.4)] * 3 + [_row("openai", "strong", 0.4)] * 1 +
        [_row("anthropic", "weak", 0.0)] * 1 + [_row("openai", "weak", 0.0)] * 3
    )
    naive = difference_in_differences(rows, "anthropic", "openai")
    controlled = tier_controlled_family_effect(rows, "anthropic", "openai")["tier_controlled_effect"]
    assert naive > 0.15                          # confounded: looks like a family effect
    assert math.isclose(controlled, 0.0, abs_tol=1e-9)   # ...but it's entirely tier


def test_stratified_permutation_is_deterministic():
    rows = (
        [_row("anthropic", "strong", 0.6)] * 6 + [_row("openai", "strong", 0.0)] * 6 +
        [_row("anthropic", "weak", 0.6)] * 6 + [_row("openai", "weak", 0.0)] * 6
    )
    r1 = stratified_permutation_test(rows, "anthropic", "openai", n_perm=2000, seed=0)
    r2 = stratified_permutation_test(rows, "anthropic", "openai", n_perm=2000, seed=0)
    assert r1["observed"] == r2["observed"] and r1["p_value"] == r2["p_value"]
    assert math.isclose(r1["observed"], 0.6, rel_tol=1e-9)
    assert r1["p_value"] < 0.05

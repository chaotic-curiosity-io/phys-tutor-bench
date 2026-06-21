"""Predictive (criterion) validity probe: does tutoring *process* predict *transfer*?

This is the in-silico dry run of the keystone validity-bridge question. The benchmark's
five process dimensions (MD, SS, ADR, CMB, PHA) describe how well the tutor *teaches*;
Transfer Success (TS) is the simulated *outcome* — whether the student then applies the
concept to a novel problem. If process quality is a valid proxy for learning, the process
composite should predict TS.

The trap: one judge scores both the process dimensions AND TS from the *same transcript*,
so a naive within-judge process<->TS correlation is inflated by shared-method variance
(one rater, one reading). The dual-judge frontier set breaks this for free: regress one
judge's process composite against the *other* judge's TS. Cross-rater correlation cannot be
a same-reading artifact. We report both, plus the range-restriction context that brackets
the truth between them.

Statistics are reused from the project's existing helpers; the novel, testable logic here is
the process composite (transfer-excluded, weights renormalized over the five) and the
cross-judge alignment.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from src.scenarios.schema import ConversationScore
from src.scoring.rubric import ALL_DIMENSIONS, DEFAULT_WEIGHTS
from src.scoring.scorecard import load_scores
from src.validation.agreement_analysis import spearman_rho

OUTCOME_DIMENSION = "transfer_success"
PROCESS_DIMENSIONS = tuple(d.id for d in ALL_DIMENSIONS if d.id != OUTCOME_DIMENSION)


# ---------------------------------------------------------------------------
# Pure, testable building blocks
# ---------------------------------------------------------------------------
def dimension_map(cs: ConversationScore) -> dict[str, int]:
    """{dimension_id -> 0-4 score} for one scored conversation."""
    return {ds.dimension: ds.score for ds in cs.scores}


def process_composite(dmap: dict[str, int], weights: dict[str, float] | None = None) -> float:
    """Weighted mean of the five process dimensions, EXCLUDING transfer_success.

    Rubric weights are renormalized over the five so the result stays on the 0-4 scale
    and is interpretable as 'process quality' independent of the outcome being predicted.
    """
    weights = weights or DEFAULT_WEIGHTS
    num = sum(weights[d] * dmap.get(d, 0) for d in PROCESS_DIMENSIONS)
    den = sum(weights[d] for d in PROCESS_DIMENSIONS)
    return num / den


def align_predictor_outcome(
    predictor_scores: list[ConversationScore],
    outcome_scores: list[ConversationScore],
) -> tuple[list[float], list[int], list[str]]:
    """Inner-join two score sets on conversation_id.

    Predictor = process composite from ``predictor_scores``; outcome = transfer_success
    from ``outcome_scores``. Pass the SAME list twice for a within-judge (inflated)
    estimate, or two different judges' scores for the cross-judge (circularity-broken) one.
    """
    pred_by = {cs.conversation_id: cs for cs in predictor_scores}
    out_by = {cs.conversation_id: cs for cs in outcome_scores}
    cids = sorted(set(pred_by) & set(out_by))
    predictors = [process_composite(dimension_map(pred_by[c])) for c in cids]
    outcomes = [dimension_map(out_by[c])[OUTCOME_DIMENSION] for c in cids]
    return predictors, outcomes, cids


# ---------------------------------------------------------------------------
# Statistics (Pearson exact; Spearman via the project's shared helper)
# ---------------------------------------------------------------------------
def pearson(x: list[float], y: list[float]) -> float | None:
    """Pearson r; None if undefined (n<2 or a constant vector / zero variance)."""
    n = len(x)
    if n < 2:
        return None
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    sx, sy = xa.std(), ya.std()
    if sx == 0 or sy == 0:
        return None
    return float(np.mean((xa - xa.mean()) * (ya - ya.mean())) / (sx * sy))


def correlate(predictors: list[float], outcomes: list[int]) -> dict:
    """Correlation block with the descriptive variance that explains attenuation."""
    return {
        "n": len(outcomes),
        "pearson": pearson(predictors, outcomes),
        "spearman": spearman_rho(predictors, outcomes) if len(outcomes) >= 2 else None,
        "predictor_std": float(np.std(predictors)) if predictors else None,
        "outcome_std": float(np.std(outcomes)) if outcomes else None,
    }


def _mean(vals: list[float | None]) -> float | None:
    real = [v for v in vals if v is not None]
    return float(np.mean(real)) if real else None


def range_restriction_correct(r: float | None, restricted_sd: float, unrestricted_sd: float) -> float | None:
    """Thorndike Case II disattenuation for direct range restriction on the predictor.

    The frontier set restricts the predictor's range (all strong tutors), attenuating r.
    Given the restricted-group SD and an estimate of the unrestricted SD, recover the
    correlation that would hold across the full range. Approximate: assumes linearity and
    homoscedasticity. Returns None if undefined.
    """
    if r is None or not restricted_sd:
        return None
    u = unrestricted_sd / restricted_sd
    denom = math.sqrt(1 - r * r + r * r * u * u)
    return (u * r) / denom if denom else None


def cross_judge(
    a_scores: list[ConversationScore],
    b_scores: list[ConversationScore],
    a_name: str,
    b_name: str,
) -> dict:
    """Both circularity-broken directions plus their mean."""
    pa, oa, _ = align_predictor_outcome(a_scores, b_scores)   # A process -> B transfer
    pb, ob, _ = align_predictor_outcome(b_scores, a_scores)   # B process -> A transfer
    d1, d2 = correlate(pa, oa), correlate(pb, ob)
    return {
        "direction_1": {"predictor_judge": a_name, "outcome_judge": b_name, **d1},
        "direction_2": {"predictor_judge": b_name, "outcome_judge": a_name, **d2},
        "mean_pearson": _mean([d1["pearson"], d2["pearson"]]),
        "mean_spearman": _mean([d1["spearman"], d2["spearman"]]),
    }


def per_dimension_cross(
    a_scores: list[ConversationScore],
    b_scores: list[ConversationScore],
) -> dict:
    """For each process dimension, mean cross-judge Pearson of (dim -> other judge's TS)."""
    a_by = {cs.conversation_id: dimension_map(cs) for cs in a_scores}
    b_by = {cs.conversation_id: dimension_map(cs) for cs in b_scores}
    cids = sorted(set(a_by) & set(b_by))
    out = {}
    for dim in PROCESS_DIMENSIONS:
        r1 = pearson([a_by[c][dim] for c in cids], [b_by[c][OUTCOME_DIMENSION] for c in cids])
        r2 = pearson([b_by[c][dim] for c in cids], [a_by[c][OUTCOME_DIMENSION] for c in cids])
        out[dim] = {"mean_pearson": _mean([r1, r2]), "n": len(cids)}
    return out


def within_judge(scores: list[ConversationScore]) -> dict:
    """Inflated single-judge estimate (predictor and outcome from the same reading).

    Predictor and outcome both come from the SAME ConversationScore object, so this maps
    each scored conversation to exactly one (process, transfer) point — no conversation_id
    join (which would silently dedupe conversations scored by more than one judge).
    """
    predictors = [process_composite(dimension_map(cs)) for cs in scores]
    outcomes = [dimension_map(cs)[OUTCOME_DIMENSION] for cs in scores]
    return correlate(predictors, outcomes)


def ts_distribution(scores: list[ConversationScore]) -> dict[int, int]:
    c = Counter(dimension_map(cs)[OUTCOME_DIMENSION] for cs in scores)
    return {k: c.get(k, 0) for k in range(5)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_bridge_analysis(opus_dir: str | Path, gpt_dir: str | Path,
                        all_results_dir: str | Path = "results") -> dict:
    """Full probe. Returns a JSON-serializable results dict."""
    opus = load_scores(opus_dir)
    gpt = load_scores(gpt_dir)
    panel = load_scores(all_results_dir)

    by_judge: dict[str, list[ConversationScore]] = {}
    for cs in panel:
        by_judge.setdefault(cs.judge_model, []).append(cs)
    # Local Ollama tutors were scored by one manual Opus-family judge (label varies slightly).
    local = [cs for j, ss in by_judge.items() if "Claude Code" in j for cs in ss]
    # Independent wide-range within-judge set: distinct conversations, Opus-family judge throughout.
    opus_family_full_range = opus + local

    def _std_process(scores):
        return float(np.std([process_composite(dimension_map(cs)) for cs in scores])) if scores else None

    cross = cross_judge(opus, gpt, "claude-opus-4-8", "gpt-5.5")
    restricted_sd = _std_process(opus)
    unrestricted_sd = _std_process(opus_family_full_range)

    return {
        "primary_cross_judge_frontier": cross,
        "bracketing": {
            "note": "Lower bound: circularity-broken but range-restricted. Upper bound: full-range "
                    "but single-judge. A range-restriction correction of the lower bound should "
                    "approach the upper bound if both estimate the same underlying relationship.",
            "cross_judge_mean_pearson_restricted": cross["mean_pearson"],
            "wide_range_within_judge_pearson": within_judge(opus_family_full_range)["pearson"],
            "cross_judge_pearson_range_corrected": range_restriction_correct(
                cross["mean_pearson"], restricted_sd or 0.0, unrestricted_sd or 0.0),
            "restricted_process_sd": restricted_sd,
            "unrestricted_process_sd": unrestricted_sd,
        },
        "per_dimension_cross_judge": per_dimension_cross(opus, gpt),
        "within_judge": {
            "frontier_opus_judge_narrow": {"n": len(opus), "range": "frontier", **within_judge(opus)},
            "frontier_gpt_judge_narrow": {"n": len(gpt), "range": "frontier", **within_judge(gpt)},
            "local_manual_judge_wide": {"n": len(local), "range": "local", **within_judge(local)},
            "opus_family_full_range": {
                "n": len(opus_family_full_range), "range": "full (local+frontier), distinct convs",
                **within_judge(opus_family_full_range),
            },
        },
        "descriptive": {
            "ts_distribution_all": ts_distribution(panel),
            "ts_zeros_all": ts_distribution(panel)[0],
            "process_std_frontier_opus_narrow": _std_process(opus),
            "process_std_opus_family_full_range": _std_process(opus_family_full_range),
            "n_panel_rows": len(panel),
            "n_distinct_conversations": len({cs.conversation_id for cs in panel}),
        },
    }

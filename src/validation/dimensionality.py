"""Construct validity & dimension redundancy: are the six dimensions really six?

PhysTutorBench scores tutors on six dimensions and reports a weighted composite. If those
dimensions are highly inter-correlated, the rubric is over-specified — it is really measuring
one "good tutoring" factor wearing six hats, and the composite weights are theatre.

The catch (same one the bridge probe and judge-bias studies hit): a within-judge correlation
matrix is inflated by single-rater HALO — one judge formed one overall impression of the
transcript and let it bleed across all six ratings — so it OVERSTATES how unidimensional the
rubric is. The clean instrument is a cross-judge multitrait-multimethod (MTMM) matrix: correlate
judge A's dimension i against judge B's dimension j. Its diagonal (A's MD vs B's MD) is each
dimension's cross-judge convergent reliability; its off-diagonal is the inter-dimension
correlation with halo removed (Campbell-Fiske discriminant logic, judges as the two methods).

We bracket: within-judge eigen/inter-correlation is an UPPER bound on unidimensionality (halo
inflates it); cross-judge is a LOWER bound (imperfect reliability attenuates it). Reuses
construct_validity's Pearson/matrix helpers and judge_bias's cross-judge alignment.
"""

from __future__ import annotations

import numpy as np

from src.scoring.rubric import ALL_DIMENSIONS, DIMENSION_BY_ID
from src.scoring.scorecard import load_scores
from src.scenarios.schema import ConversationScore
from src.validation.construct_validity import (
    pearson_correlation,
    correlation_matrix,
    _extract_dimension_arrays,
)
from src.validation.judge_bias import align_judges

DIM_IDS = [d.id for d in ALL_DIMENSIONS]
ABBR = {d.id: d.abbreviation for d in ALL_DIMENSIONS}


def cronbach_alpha(scores: list[ConversationScore]) -> float | None:
    """Internal consistency of the six dimensions treated as items of one scale.

    None if fewer than two conversations or the total has no variance.
    """
    arrays = _extract_dimension_arrays(scores)
    items = np.array([arrays[d] for d in DIM_IDS], dtype=float)   # (6, n)
    k, n = items.shape
    if n < 2:
        return None
    item_var = items.var(axis=1, ddof=1).sum()
    total_var = items.sum(axis=0).var(ddof=1)
    if total_var == 0:
        return None
    return float(k / (k - 1) * (1 - item_var / total_var))


def eigen_summary(corr: np.ndarray) -> dict:
    """Eigenvalues of a correlation matrix -> dimensionality. pc1_frac = share of the first."""
    w = np.sort(np.linalg.eigvalsh(corr))[::-1]
    total = float(w.sum())
    return {
        "eigenvalues": [float(x) for x in w],
        "pc1_frac": float(w[0] / total) if total else None,
        "kaiser_factors": int(np.sum(w > 1.0 + 1e-9)),   # eigenvalues strictly > 1
    }


def mean_offdiag(corr: np.ndarray) -> float:
    """Mean absolute off-diagonal correlation (upper triangle)."""
    n = corr.shape[0]
    vals = [abs(corr[i][j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals)) if vals else 0.0


def cross_judge_mtmm(a_scores: list[ConversationScore],
                     b_scores: list[ConversationScore]) -> tuple[np.ndarray, int]:
    """MTMM matrix M[i][j] = corr(judge A's dim i, judge B's dim j) over shared conversations."""
    rows = align_judges(a_scores, b_scores)
    M = np.zeros((len(DIM_IDS), len(DIM_IDS)))
    for i, di in enumerate(DIM_IDS):
        ai = [r["a_dims"].get(di, 0) for r in rows]
        for j, dj in enumerate(DIM_IDS):
            bj = [r["b_dims"].get(dj, 0) for r in rows]
            M[i][j] = pearson_correlation(ai, bj)
    return M, len(rows)


def mtmm_summary(M: np.ndarray) -> dict:
    """Convergent (diagonal) vs heterotrait (off-diagonal) + Campbell-Fiske violation count."""
    n = M.shape[0]
    diag = [M[i][i] for i in range(n)]
    off = [M[i][j] for i in range(n) for j in range(n) if i != j]
    violations = comparisons = 0
    for i in range(n):
        conv = M[i][i]
        for j in range(n):
            if i == j:
                continue
            comparisons += 2
            violations += int(conv <= M[i][j]) + int(conv <= M[j][i])
    return {
        "convergent_diagonal": {ABBR[DIM_IDS[i]]: float(M[i][i]) for i in range(n)},
        "mean_convergent": float(np.mean(diag)),
        "mean_heterotrait": float(np.mean(np.abs(off))),
        "campbell_fiske_violations": violations,
        "campbell_fiske_comparisons": comparisons,
    }


def _symmetrize(M: np.ndarray) -> np.ndarray:
    S = (M + M.T) / 2
    np.fill_diagonal(S, 1.0)
    return S


def redundancy_pairs(corr: np.ndarray, threshold: float = 0.8) -> list[tuple[str, str, float]]:
    """Dimension pairs with |r| >= threshold (candidates to merge), strongest first."""
    n = corr.shape[0]
    pairs = [(ABBR[DIM_IDS[i]], ABBR[DIM_IDS[j]], float(corr[i][j]))
             for i in range(n) for j in range(i + 1, n) if abs(corr[i][j]) >= threshold]
    return sorted(pairs, key=lambda t: -abs(t[2]))


def run_dimensionality_analysis(opus_dir, gpt_dir, all_results_dir="results") -> dict:
    """Full construct-validity / redundancy probe. Returns a JSON-serializable dict."""
    opus = load_scores(opus_dir)
    gpt = load_scores(gpt_dir)
    panel = load_scores(all_results_dir)

    m_all, dim_ids = correlation_matrix(panel)
    m_front, _ = correlation_matrix(opus)             # within-judge, frontier convs (halo, narrow)
    M, n_cross = cross_judge_mtmm(opus, gpt)          # cross-judge, frontier convs (halo-free)
    S = _symmetrize(M)
    mt = mtmm_summary(M)
    labels = [ABBR[d] for d in dim_ids]

    return {
        "n": {"within_pooled": len(panel), "within_frontier": len(opus), "cross_frontier": n_cross},
        "within_pooled": {
            "labels": labels,
            "matrix": m_all.tolist(),
            "mean_offdiag": mean_offdiag(m_all),
            "cronbach_alpha": cronbach_alpha(panel),
            "eigen": eigen_summary(m_all),
            "redundancy_pairs_0.8": redundancy_pairs(m_all, 0.8),
        },
        "cross_frontier": {
            "labels": labels,
            "mtmm_matrix": M.tolist(),
            **mt,
            "eigen_symmetrized": eigen_summary(S),
            "mean_offdiag_symmetrized": mean_offdiag(S),
            "redundancy_pairs_0.8": redundancy_pairs(S, 0.8),
        },
        "halo": {
            "within_frontier_mean_offdiag": mean_offdiag(m_front),
            "cross_frontier_mean_heterotrait": mt["mean_heterotrait"],
            "halo_inflation": mean_offdiag(m_front) - mt["mean_heterotrait"],
            "note": "Same 48 frontier conversations both ways; the drop from within- to cross-judge "
                    "inter-dimension correlation is single-rater halo (method variance).",
        },
    }

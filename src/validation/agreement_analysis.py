"""Inter-rater reliability analysis for human annotations."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def load_annotations_from_db(db_path: str | Path) -> list[dict]:
    """Load all annotations from the SQLite database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT * FROM annotations ORDER BY conversation_id, dimension, annotator_id")
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return rows


def cohens_kappa(rater1: list[int], rater2: list[int], k: int = 5) -> float:
    """Compute Cohen's kappa for two raters on ordinal data.

    Args:
        rater1: Scores from rater 1
        rater2: Scores from rater 2
        k: Number of categories (0-4 = 5 categories)
    """
    n = len(rater1)
    if n == 0:
        return 0.0

    # Build confusion matrix
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(rater1, rater2):
        matrix[a][b] += 1

    # Observed agreement
    p_o = np.sum(np.diag(matrix)) / n

    # Expected agreement
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    p_e = np.sum(row_sums * col_sums)

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)


def weighted_cohens_kappa(rater1: list[int], rater2: list[int], k: int = 5) -> float:
    """Compute quadratic-weighted Cohen's kappa (better for ordinal data)."""
    n = len(rater1)
    if n == 0:
        return 0.0

    matrix = np.zeros((k, k), dtype=float)
    for a, b in zip(rater1, rater2):
        matrix[a][b] += 1

    # Weight matrix (quadratic)
    weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            weights[i][j] = (i - j) ** 2 / (k - 1) ** 2

    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    expected = np.outer(row_sums, col_sums)

    observed_weighted = np.sum(weights * matrix / n)
    expected_weighted = np.sum(weights * expected)

    if expected_weighted == 0:
        return 1.0

    return 1.0 - observed_weighted / expected_weighted


def krippendorff_alpha(data: list[list[int | None]], k: int = 5) -> float:
    """Compute Krippendorff's alpha for multiple annotators.

    Args:
        data: Matrix where data[annotator][item] = score (None if not rated).
        k: Number of categories.
    """
    n_annotators = len(data)
    if n_annotators < 2:
        return 0.0

    n_items = len(data[0]) if data else 0
    if n_items == 0:
        return 0.0

    # Collect value pairs for observed disagreement
    observed_pairs: list[tuple[int, int]] = []
    all_values: list[int] = []

    for item_idx in range(n_items):
        item_values = []
        for ann_idx in range(n_annotators):
            val = data[ann_idx][item_idx] if item_idx < len(data[ann_idx]) else None
            if val is not None:
                item_values.append(val)
                all_values.append(val)

        # Generate all pairs within this item
        for i in range(len(item_values)):
            for j in range(i + 1, len(item_values)):
                observed_pairs.append((item_values[i], item_values[j]))

    if not observed_pairs or not all_values:
        return 0.0

    # Observed disagreement (ordinal distance)
    d_o = np.mean([(a - b) ** 2 for a, b in observed_pairs])

    # Expected disagreement
    n_total = len(all_values)
    expected_pairs = []
    for i in range(n_total):
        for j in range(i + 1, n_total):
            expected_pairs.append((all_values[i], all_values[j]))

    if not expected_pairs:
        return 0.0

    d_e = np.mean([(a - b) ** 2 for a, b in expected_pairs])

    if d_e == 0:
        return 1.0

    return 1.0 - d_o / d_e


def spearman_rho(x: list[int], y: list[int]) -> float:
    """Compute Spearman's rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    # Rank the data
    def rank(values: list[int]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and values[sorted_indices[j]] == values[sorted_indices[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1
            for idx in range(i, j):
                ranks[sorted_indices[idx]] = avg_rank
            i = j
        return ranks

    rx = rank(x)
    ry = rank(y)

    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6 * d_sq) / (n * (n ** 2 - 1))


def confusion_matrix(rater1: list[int], rater2: list[int], k: int = 5) -> np.ndarray:
    """Build a k x k confusion matrix."""
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(rater1, rater2):
        matrix[a][b] += 1
    return matrix


def run_agreement_analysis(db_path: str | Path) -> dict:
    """Run full inter-rater reliability analysis.

    Returns a dict with results for each dimension and overall.
    """
    annotations = load_annotations_from_db(db_path)

    if not annotations:
        console.print("[yellow]No annotations found.[/yellow]")
        return {}

    # Organize by (conversation_id, dimension) -> {annotator_id: score}
    score_map: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for ann in annotations:
        key = (ann["conversation_id"], ann["dimension"])
        score_map[key][ann["annotator_id"]] = ann["score"]

    # Get all annotators
    all_annotators = sorted(set(ann["annotator_id"] for ann in annotations))
    dimensions = sorted(set(ann["dimension"] for ann in annotations))

    results: dict[str, dict] = {}

    for dim in dimensions:
        # Get pairs of annotators' scores for this dimension
        dim_items = {k: v for k, v in score_map.items() if k[1] == dim}

        # Pairwise Cohen's kappa
        pairwise_kappas = {}
        for i in range(len(all_annotators)):
            for j in range(i + 1, len(all_annotators)):
                a1, a2 = all_annotators[i], all_annotators[j]
                r1, r2 = [], []
                for key, scores in dim_items.items():
                    if a1 in scores and a2 in scores:
                        r1.append(scores[a1])
                        r2.append(scores[a2])
                if r1:
                    kappa = weighted_cohens_kappa(r1, r2)
                    rho = spearman_rho(r1, r2)
                    pairwise_kappas[f"{a1}_vs_{a2}"] = {
                        "kappa": kappa,
                        "spearman_rho": rho,
                        "n_items": len(r1),
                    }

        # Krippendorff's alpha (all annotators)
        items = sorted(set(k[0] for k in dim_items))
        data_matrix = []
        for ann_id in all_annotators:
            row = []
            for item in items:
                scores = dim_items.get((item, dim), {})
                row.append(scores.get(ann_id))
            data_matrix.append(row)

        alpha = krippendorff_alpha(data_matrix)

        results[dim] = {
            "krippendorff_alpha": alpha,
            "pairwise": pairwise_kappas,
            "n_items": len(items),
            "n_annotators": len(all_annotators),
        }

    # Print results
    table = Table(title="Inter-Rater Reliability Analysis")
    table.add_column("Dimension")
    table.add_column("Krippendorff's Alpha", justify="right")
    table.add_column("N Items", justify="right")
    table.add_column("N Annotators", justify="right")

    for dim, data in results.items():
        alpha = data["krippendorff_alpha"]
        quality = "good" if alpha > 0.6 else "fair" if alpha > 0.4 else "poor"
        table.add_row(
            dim,
            f"{alpha:.3f} ({quality})",
            str(data["n_items"]),
            str(data["n_annotators"]),
        )

    console.print(table)
    return results

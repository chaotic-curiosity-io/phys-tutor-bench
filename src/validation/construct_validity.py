"""Construct validity analysis — verifying the rubric dimensions measure distinct, meaningful constructs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from src.scoring.rubric import ALL_DIMENSIONS
from src.scoring.scorecard import load_scores
from src.scenarios.schema import ConversationScore

console = Console()


def _extract_dimension_arrays(scores: list[ConversationScore]) -> dict[str, list[int]]:
    """Extract per-dimension score arrays from conversation scores."""
    arrays: dict[str, list[int]] = {dim.id: [] for dim in ALL_DIMENSIONS}
    for cs in scores:
        dim_map = {ds.dimension: ds.score for ds in cs.scores}
        for dim in ALL_DIMENSIONS:
            arrays[dim.id].append(dim_map.get(dim.id, 0))
    return arrays


def pearson_correlation(x: list[int], y: list[int]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    mx, my = xa.mean(), ya.mean()
    sx, sy = xa.std(), ya.std()
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.mean((xa - mx) * (ya - my)) / (sx * sy))


def correlation_matrix(scores: list[ConversationScore]) -> tuple[np.ndarray, list[str]]:
    """Compute the correlation matrix across all dimensions.

    Returns:
        Tuple of (correlation matrix, dimension labels)
    """
    arrays = _extract_dimension_arrays(scores)
    dim_ids = [dim.id for dim in ALL_DIMENSIONS]

    n = len(dim_ids)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i][j] = pearson_correlation(arrays[dim_ids[i]], arrays[dim_ids[j]])

    return matrix, dim_ids


def run_construct_validity(scores: list[ConversationScore]) -> dict:
    """Run construct validity analyses.

    Tests key hypotheses:
    1. MD <-> SS: Correct diagnosis should correlate with better scaffolding
    2. ADR <-> TS: Answer restraint should predict transfer success
    3. CMB <-> TS: Conceptual model building should predict transfer
    4. Factor analysis: Dimensions should measure distinct constructs

    Returns dict with analysis results.
    """
    arrays = _extract_dimension_arrays(scores)
    results: dict = {}

    # Hypothesis 1: MD correlates with SS
    r_md_ss = pearson_correlation(arrays["misconception_diagnosis"], arrays["scaffolding_strategy"])
    results["md_ss_correlation"] = {
        "r": r_md_ss,
        "hypothesis": "Correct diagnosis correlates with better scaffolding",
        "interpretation": _interpret_correlation(r_md_ss),
    }

    # Hypothesis 2: ADR correlates with TS
    r_adr_ts = pearson_correlation(arrays["answer_disclosure_restraint"], arrays["transfer_success"])
    results["adr_ts_correlation"] = {
        "r": r_adr_ts,
        "hypothesis": "Answer restraint predicts transfer success",
        "interpretation": _interpret_correlation(r_adr_ts),
    }

    # Hypothesis 3: CMB correlates with TS
    r_cmb_ts = pearson_correlation(arrays["conceptual_model_building"], arrays["transfer_success"])
    results["cmb_ts_correlation"] = {
        "r": r_cmb_ts,
        "hypothesis": "Conceptual model building predicts transfer",
        "interpretation": _interpret_correlation(r_cmb_ts),
    }

    # Full correlation matrix
    corr_matrix, dim_ids = correlation_matrix(scores)
    results["correlation_matrix"] = {
        "matrix": corr_matrix.tolist(),
        "dimensions": dim_ids,
    }

    # Check discriminability — average off-diagonal correlation should be moderate, not high
    n = len(dim_ids)
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(abs(corr_matrix[i][j]))
    avg_off_diag = float(np.mean(off_diag)) if off_diag else 0.0
    results["avg_inter_dimension_correlation"] = avg_off_diag
    results["dimensions_distinct"] = avg_off_diag < 0.7

    # Print results
    console.print("\n[bold]Construct Validity Analysis[/bold]\n")

    # Hypothesis table
    table = Table(title="Key Hypotheses")
    table.add_column("Hypothesis")
    table.add_column("r", justify="right")
    table.add_column("Interpretation")

    for key in ["md_ss_correlation", "adr_ts_correlation", "cmb_ts_correlation"]:
        data = results[key]
        table.add_row(data["hypothesis"], f"{data['r']:.3f}", data["interpretation"])

    console.print(table)

    # Correlation matrix table
    dim_labels = [dim.abbreviation for dim in ALL_DIMENSIONS]
    corr_table = Table(title="Dimension Correlation Matrix")
    corr_table.add_column("")
    for label in dim_labels:
        corr_table.add_column(label, justify="right")

    for i, label in enumerate(dim_labels):
        row = [label]
        for j in range(len(dim_labels)):
            val = corr_matrix[i][j]
            row.append(f"{val:.2f}")
        corr_table.add_row(*row)

    console.print(corr_table)

    console.print(f"\nAverage inter-dimension correlation: {avg_off_diag:.3f}")
    if results["dimensions_distinct"]:
        console.print("[green]Dimensions appear to measure distinct constructs.[/green]")
    else:
        console.print("[yellow]Warning: High inter-dimension correlation suggests some dimensions may overlap.[/yellow]")

    return results


def _interpret_correlation(r: float) -> str:
    """Interpret a Pearson correlation value."""
    ar = abs(r)
    if ar >= 0.7:
        strength = "strong"
    elif ar >= 0.4:
        strength = "moderate"
    elif ar >= 0.2:
        strength = "weak"
    else:
        strength = "negligible"

    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction} ({r:.3f})"

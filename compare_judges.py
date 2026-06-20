"""Inter-judge agreement between two judges that scored the same conversations.

Reads two directories of ``*_score.json`` files (one per judge) and reports, per
rubric dimension, how well the two judges agree: quadratic-weighted Cohen's
kappa, unweighted kappa, Krippendorff's alpha (2 raters), Spearman's rho, exact
agreement, and within-one agreement. Also reports composite-score agreement.

Reuses the statistics in ``src/validation/agreement_analysis.py`` — the same
machinery used for human-vs-LLM agreement.

Usage:
    python compare_judges.py --a results/frontier/_scores/opus-4-8 \
                             --b results/frontier/_scores/gpt-5.5 \
                             --output results/frontier/judge_agreement.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.scoring.scorecard import load_scores
from src.validation.agreement_analysis import (
    cohens_kappa,
    weighted_cohens_kappa,
    krippendorff_alpha,
    spearman_rho,
)

console = Console()

# Canonical dimension order (others, if any, are appended alphabetically).
_DIM_ORDER = [
    "misconception_diagnosis",
    "scaffolding_strategy",
    "answer_disclosure_restraint",
    "conceptual_model_building",
    "transfer_success",
    "pedagogical_harm_avoidance",
]


def _index(scores):
    """conversation_id -> {dimension: score}, plus a composite map."""
    by_dim: dict[str, dict[str, int]] = {}
    composite: dict[str, float] = {}
    for s in scores:
        by_dim[s.conversation_id] = {ds.dimension: ds.score for ds in s.scores}
        composite[s.conversation_id] = s.composite_score
    return by_dim, composite


def _ordered_dims(present: set[str]) -> list[str]:
    ordered = [d for d in _DIM_ORDER if d in present]
    ordered += sorted(present - set(ordered))
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser(description="Inter-judge agreement report")
    ap.add_argument("--a", required=True, help="Score dir for judge A")
    ap.add_argument("--b", required=True, help="Score dir for judge B")
    ap.add_argument("--label-a", default=None, help="Display label for judge A")
    ap.add_argument("--label-b", default=None, help="Display label for judge B")
    ap.add_argument("--output", default=None, help="Optional path to write JSON report")
    args = ap.parse_args()

    scores_a = load_scores(args.a)
    scores_b = load_scores(args.b)
    if not scores_a or not scores_b:
        console.print("[red]One or both judge score sets are empty.[/red]")
        raise SystemExit(1)

    label_a = args.label_a or (scores_a[0].judge_model if scores_a else "judge_a")
    label_b = args.label_b or (scores_b[0].judge_model if scores_b else "judge_b")

    a_dim, a_comp = _index(scores_a)
    b_dim, b_comp = _index(scores_b)
    common = sorted(set(a_dim) & set(b_dim))
    if not common:
        console.print("[red]No conversations scored by both judges.[/red]")
        raise SystemExit(1)

    present_dims: set[str] = set()
    for cid in common:
        present_dims |= set(a_dim[cid]) & set(b_dim[cid])
    dims = _ordered_dims(present_dims)

    results: dict[str, dict] = {}
    for dim in dims:
        r1, r2 = [], []
        for cid in common:
            if dim in a_dim[cid] and dim in b_dim[cid]:
                r1.append(int(a_dim[cid][dim]))
                r2.append(int(b_dim[cid][dim]))
        if not r1:
            continue
        n = len(r1)
        results[dim] = {
            "n": n,
            "weighted_kappa": weighted_cohens_kappa(r1, r2),
            "cohens_kappa": cohens_kappa(r1, r2),
            "krippendorff_alpha": krippendorff_alpha([r1, r2]),
            "spearman_rho": spearman_rho(r1, r2),
            "exact_agreement": sum(x == y for x, y in zip(r1, r2)) / n,
            "within_one": sum(abs(x - y) <= 1 for x, y in zip(r1, r2)) / n,
            "mean_a": sum(r1) / n,
            "mean_b": sum(r2) / n,
        }

    # Composite-score agreement (Spearman on rounded composites + mean abs diff).
    ca = [a_comp[c] for c in common]
    cb = [b_comp[c] for c in common]
    composite = {
        "n": len(common),
        "spearman_rho": spearman_rho(
            [round(x * 10) for x in ca], [round(x * 10) for x in cb]
        ),
        "mean_abs_diff": sum(abs(x - y) for x, y in zip(ca, cb)) / len(common),
        "mean_a": sum(ca) / len(common),
        "mean_b": sum(cb) / len(common),
    }

    report = {
        "judge_a": label_a,
        "judge_b": label_b,
        "n_conversations": len(common),
        "per_dimension": results,
        "composite": composite,
    }
    # Write JSON first so the data is saved even if console rendering fails.
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))

    # ASCII-only output (Windows consoles are cp1252 and choke on greek/em-dash).
    table = Table(title=f"Inter-Judge Agreement: {label_a} vs {label_b} (n={len(common)})")
    table.add_column("Dimension")
    table.add_column("WtdKappa", justify="right")
    table.add_column("Kappa", justify="right")
    table.add_column("KrippAlpha", justify="right")
    table.add_column("Spearman", justify="right")
    table.add_column("Exact", justify="right")
    table.add_column("Within1", justify="right")
    table.add_column(f"mean[{label_a[:8]}]", justify="right")
    table.add_column(f"mean[{label_b[:8]}]", justify="right")
    for dim, r in results.items():
        wk = r["weighted_kappa"]
        quality = "good" if wk > 0.6 else "fair" if wk > 0.4 else "poor"
        table.add_row(
            dim,
            f"{wk:.3f} ({quality})",
            f"{r['cohens_kappa']:.3f}",
            f"{r['krippendorff_alpha']:.3f}",
            f"{r['spearman_rho']:.3f}",
            f"{r['exact_agreement']:.0%}",
            f"{r['within_one']:.0%}",
            f"{r['mean_a']:.2f}",
            f"{r['mean_b']:.2f}",
        )
    try:
        console.print(table)
        console.print(
            f"\n[bold]Composite[/bold]: Spearman rho={composite['spearman_rho']:.3f}, "
            f"mean|diff|={composite['mean_abs_diff']:.3f}, "
            f"mean[{label_a}]={composite['mean_a']:.2f}, mean[{label_b}]={composite['mean_b']:.2f}"
        )
    except UnicodeEncodeError:
        # Last-resort plain rendering for legacy terminals.
        print(f"Inter-Judge Agreement: {label_a} vs {label_b} (n={len(common)})")
        for dim, r in results.items():
            print(f"  {dim:<28} wtdK={r['weighted_kappa']:.3f} kappa={r['cohens_kappa']:.3f} "
                  f"krippA={r['krippendorff_alpha']:.3f} spearman={r['spearman_rho']:.3f} "
                  f"exact={r['exact_agreement']:.0%} within1={r['within_one']:.0%}")
        print(f"  Composite: spearman={composite['spearman_rho']:.3f} "
              f"mean|diff|={composite['mean_abs_diff']:.3f}")
    if args.output:
        console.print(f"\n[green]Wrote[/green] {args.output}")


if __name__ == "__main__":
    main()

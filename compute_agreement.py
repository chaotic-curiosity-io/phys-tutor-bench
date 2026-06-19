"""Inter-rater reliability between independent judge passes.

Each judge pass writes raw-score JSON per conversation under
results/<model>/<run>/<raw_dirname>/<conversation_id>.json (see finalize_scores.py
for the shape). This script aligns two or more passes by conversation_id and reports,
per rubric dimension, the agreement between them using the project's existing PER
agreement statistics (src/validation/agreement_analysis.py):

  - quadratic-weighted Cohen's kappa  (ordinal, pairwise pass A vs B)
  - Spearman rank correlation         (pairwise)
  - Krippendorff's alpha              (across all supplied passes)
  - % exact match and % within 1 point

Because all passes are produced by the same underlying judge model, this measures the
judge's *self-consistency* (test-retest reliability), not independent human agreement.

Usage:  python compute_agreement.py [results_root] [passA_dir] [passB_dir ...]
Default: results raw_scores raw_scores_b
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

from src.validation.agreement_analysis import (
    weighted_cohens_kappa,
    spearman_rho,
    krippendorff_alpha,
)
from src.scoring.rubric import ALL_DIMENSIONS

DIMS = [d.id for d in ALL_DIMENSIONS]
ABBR = {d.id: d.abbreviation for d in ALL_DIMENSIONS}


def load_pass(results_root: str, raw_dirname: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for f in glob.glob(f"{results_root}/*/*/{raw_dirname}/*.json"):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        cid = d["conversation_id"]
        out[cid] = {dim: int(d["dimensions"][dim]["score"]) for dim in DIMS}
    return out


def main(results_root: str = "results", passes: tuple[str, ...] = ("raw_scores", "raw_scores_b")) -> None:
    loaded = [load_pass(results_root, p) for p in passes]
    for p, l in zip(passes, loaded):
        print(f"  pass '{p}': {len(l)} conversations")
    common = set(loaded[0])
    for l in loaded[1:]:
        common &= set(l)
    common = sorted(common)
    print(f"common conversations across all passes: {len(common)}\n")
    if not common:
        print("No overlap between passes — nothing to compare.")
        return

    per_dim = {}
    for dim in DIMS:
        raters = [[l[cid][dim] for cid in common] for l in loaded]
        a, b = raters[0], raters[1]
        per_dim[dim] = {
            "weighted_kappa": weighted_cohens_kappa(a, b),
            "spearman": spearman_rho(a, b),
            "krippendorff_alpha": krippendorff_alpha(raters),
            "pct_exact": float(np.mean([x == y for x, y in zip(a, b)])),
            "pct_within_1": float(np.mean([abs(x - y) <= 1 for x, y in zip(a, b)])),
            "mean_pass_a": float(np.mean(a)),
            "mean_pass_b": float(np.mean(b)),
        }

    allA = [loaded[0][cid][dim] for dim in DIMS for cid in common]
    allB = [loaded[1][cid][dim] for dim in DIMS for cid in common]
    overall = {
        "weighted_kappa": weighted_cohens_kappa(allA, allB),
        "spearman": spearman_rho(allA, allB),
        "krippendorff_alpha": krippendorff_alpha([allA, allB]),
        "pct_exact": float(np.mean([x == y for x, y in zip(allA, allB)])),
        "pct_within_1": float(np.mean([abs(x - y) <= 1 for x, y in zip(allA, allB)])),
    }

    out = {"n_conversations": len(common), "passes": list(passes),
           "per_dimension": per_dim, "overall": overall}
    Path("docs/judge_agreement.json").write_text(json.dumps(out, indent=2))

    print(f"{'dimension':28s} {'wkappa':>7} {'rho':>6} {'alpha':>6} {'%exact':>7} {'%<=1':>6}")
    print("-" * 66)
    for dim in DIMS:
        r = per_dim[dim]
        print(f"{ABBR[dim]+' '+dim:28s} {r['weighted_kappa']:7.2f} {r['spearman']:6.2f} "
              f"{r['krippendorff_alpha']:6.2f} {r['pct_exact']*100:6.0f}% {r['pct_within_1']*100:5.0f}%")
    print("-" * 66)
    print(f"{'OVERALL (pooled)':28s} {overall['weighted_kappa']:7.2f} {overall['spearman']:6.2f} "
          f"{overall['krippendorff_alpha']:6.2f} {overall['pct_exact']*100:6.0f}% {overall['pct_within_1']*100:5.0f}%")
    print("\nwrote docs/judge_agreement.json")


if __name__ == "__main__":
    args = sys.argv[1:]
    root = args[0] if args else "results"
    ps = tuple(args[1:]) if len(args) > 1 else ("raw_scores", "raw_scores_b")
    main(root, ps)

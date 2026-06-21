"""In-silico validity-bridge probe: does tutoring PROCESS predict TRANSFER?

Runs src.validation.predictive_validity on the saved scores, prints a summary, and writes
docs/bridge_probe.json (+ docs/assets/bridge_scatter.png). No new API calls.

Usage:
    python bridge_probe.py [opus_scores_dir] [gpt_scores_dir] [all_results_dir]
Defaults to the frontier dual-judge run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.scoring.rubric import DIMENSION_BY_ID
from src.validation.predictive_validity import (
    run_bridge_analysis,
    align_predictor_outcome,
    PROCESS_DIMENSIONS,
)
from src.scoring.scorecard import load_scores

OPUS = sys.argv[1] if len(sys.argv) > 1 else "results/frontier/_scores/opus-4-8"
GPT = sys.argv[2] if len(sys.argv) > 2 else "results/frontier/_scores/gpt-5.5"
ALL = sys.argv[3] if len(sys.argv) > 3 else "results"


def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v:+.2f}"


def main() -> None:
    res = run_bridge_analysis(OPUS, GPT, ALL)

    print("\n=== PRIMARY: cross-judge process -> transfer (circularity-broken, frontier) ===")
    cj = res["primary_cross_judge_frontier"]
    for key in ("direction_1", "direction_2"):
        d = cj[key]
        print(f"  {d['predictor_judge']:>18} process -> {d['outcome_judge']:<10} TS "
              f"| n={d['n']:>3}  Pearson {_fmt(d['pearson'])}  Spearman {_fmt(d['spearman'])}  "
              f"(proc_sd={_fmt(d['predictor_std'])}, ts_sd={_fmt(d['outcome_std'])})")
    print(f"  MEAN  Pearson {_fmt(cj['mean_pearson'])}   Spearman {_fmt(cj['mean_spearman'])}")

    br = res["bracketing"]
    print("\n=== Bracketing the true relationship ===")
    print(f"  lower (cross-judge, range-restricted) Pearson {_fmt(br['cross_judge_mean_pearson_restricted'])}")
    print(f"  upper (full-range, single-judge)      Pearson {_fmt(br['wide_range_within_judge_pearson'])}")
    print(f"  lower CORRECTED for range restriction Pearson {_fmt(br['cross_judge_pearson_range_corrected'])} "
          f"(proc SD {_fmt(br['restricted_process_sd'])} -> {_fmt(br['unrestricted_process_sd'])})")

    print("\n=== Per-dimension -> transfer (mean cross-judge Pearson) ===")
    pdim = res["per_dimension_cross_judge"]
    for dim, blk in sorted(pdim.items(), key=lambda kv: (kv[1]["mean_pearson"] or -9), reverse=True):
        abbr = DIMENSION_BY_ID[dim].abbreviation
        print(f"  {abbr:<4} {dim:<28} Pearson {_fmt(blk['mean_pearson'])}  (n={blk['n']})")

    print("\n=== Within-judge process -> transfer (SAME reading; shared-method INFLATED) ===")
    for label, blk in res["within_judge"].items():
        print(f"  {label:<34} n={blk['n']:>3}  Pearson {_fmt(blk['pearson'])}  "
              f"Spearman {_fmt(blk['spearman'])}  (proc_sd={_fmt(blk['predictor_std'])})")

    print("\n=== Descriptive (caveats) ===")
    desc = res["descriptive"]
    print(f"  TS distribution ({desc['n_panel_rows']} rows / {desc['n_distinct_conversations']} "
          f"distinct convs): {desc['ts_distribution_all']}  "
          f"-> only {desc['ts_zeros_all']} 'not-reached/failed' zeros")
    print(f"  Process-composite SD: frontier-narrow {_fmt(desc['process_std_frontier_opus_narrow'])} "
          f"vs full-range {_fmt(desc['process_std_opus_family_full_range'])}  (range restriction)")

    # ---- scatter: predictor (opus process) vs outcome (gpt TS), the headline direction ----
    opus, gpt = load_scores(OPUS), load_scores(GPT)
    px, py, _ = align_predictor_outcome(opus, gpt)
    qx, qy, _ = align_predictor_outcome(gpt, opus)
    assets = Path("docs/assets")
    assets.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    jitter = lambda v, s=0.06: [y + s * ((i % 7) - 3) / 3 for i, y in enumerate(v)]
    ax.scatter(px, jitter(py), s=42, alpha=0.7, label="Opus process → GPT-5.5 TS", color="#2563eb")
    ax.scatter(qx, jitter(qy), s=42, alpha=0.7, marker="^",
               label="GPT-5.5 process → Opus TS", color="#dc2626")
    ax.set_xlabel("Tutoring process composite (predictor judge, 0–4)")
    ax.set_ylabel("Transfer Success (outcome judge, 0–4; jittered)")
    ax.set_title("Cross-judge process → transfer (circularity-broken)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    scatter_path = assets / "bridge_scatter.png"
    fig.savefig(scatter_path, dpi=130)
    print(f"\n  scatter -> {scatter_path}")

    out = Path("docs/bridge_probe.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"  results -> {out}")


if __name__ == "__main__":
    main()

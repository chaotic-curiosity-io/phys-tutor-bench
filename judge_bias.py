"""Judge self-preference probe: does each judge over-reward its own model family?

Runs src.validation.judge_bias on the frontier dual-judge scores, prints the tables, and
writes docs/judge_bias.json (+ docs/assets/judge_bias.png). No new API calls.

Usage:
    python judge_bias.py [judge_a_dir] [judge_b_dir] [a_name] [b_name]
Defaults to the frontier run (Opus 4.8 vs GPT-5.5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.scoring.rubric import DIMENSION_BY_ID
from src.validation.judge_bias import run_judge_bias_analysis, align_judges, ceiling_rates
from src.scoring.scorecard import load_scores

A_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/frontier/_scores/opus-4-8"
B_DIR = sys.argv[2] if len(sys.argv) > 2 else "results/frontier/_scores/gpt-5.5"
A_NAME = sys.argv[3] if len(sys.argv) > 3 else "claude-opus-4-8"
B_NAME = sys.argv[4] if len(sys.argv) > 4 else "gpt-5.5"


def _f(v, p=3):
    return " n/a" if v is None else f"{v:+.{p}f}"


def main() -> None:
    res = run_judge_bias_analysis(A_DIR, B_DIR, A_NAME, B_NAME)
    ja, jb = res["judges"]["a"], res["judges"]["b"]
    fa, fb = res["judges"]["a_family"], res["judges"]["b_family"]

    print(f"\nJudge A = {ja} ({fa})   Judge B = {jb} ({fb})   n = {res['n_conversations']} conversations\n")

    print("Mean composite by judge x tutor:")
    cm = res["cell_means"]
    tutors = sorted(cm["a"])
    print(f"  {'tutor':<20} {'A':>7} {'B':>7} {'A-B':>8}")
    for t in tutors:
        print(f"  {t:<20} {cm['a'][t]:7.3f} {cm['b'][t]:7.3f} {cm['a'][t]-cm['b'][t]:+8.3f}")

    print("\nPaired delta (A - B) by tutor family:")
    for f, blk in res["paired_delta_by_family"].items():
        print(f"  {f:<10} n={blk['n']:>2}  mean={_f(blk['mean'])}  sd={blk['sd']:.3f}")

    ci = res["did_bootstrap_ci95"]
    print(f"\nSELF-PREFERENCE DiD (own-family inflation, composite) = {_f(res['composite_did'])} points")
    print(f"  permutation p = {res['did_permutation_p']:.4f}   bootstrap 95% CI = "
          f"[{_f(ci[0]) if ci else 'n/a'}, {_f(ci[1]) if ci else 'n/a'}]")

    print("\nPer-dimension DiD (own-family inflation):")
    for dim, v in sorted(res["per_dimension_did"].items(), key=lambda kv: -(kv[1] or -9)):
        print(f"  {DIMENSION_BY_ID[dim].abbreviation:<4} {dim:<28} {_f(v)}")

    print("\nCeiling: % of dimension scores == 4, by judge x family:")
    cr = res["ceiling_rates"]
    for fam in sorted(set(cr["a"]) | set(cr["b"])):
        a = cr["a"].get(fam); b = cr["b"].get(fam)
        print(f"  {fam:<10}  A {100*a:5.1f}%   B {100*b:5.1f}%" if a is not None and b is not None
              else f"  {fam:<10}  (incomplete)")

    _figure(res)

    out = Path("docs/judge_bias.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"\n  results -> {out}")


def _figure(res: dict) -> None:
    cm = res["cell_means"]
    cr = res["ceiling_rates"]
    tutors = sorted(cm["a"])
    ja, jb = res["judges"]["a"], res["judges"]["b"]
    assets = Path("docs/assets"); assets.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(tutors)); w = 0.38
    ax1.bar([i - w / 2 for i in x], [cm["a"][t] for t in tutors], w, label=f"{ja} judge", color="#2563eb")
    ax1.bar([i + w / 2 for i in x], [cm["b"][t] for t in tutors], w, label=f"{jb} judge", color="#dc2626")
    ax1.set_xticks(list(x)); ax1.set_xticklabels([t.replace("claude-", "") for t in tutors], rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("Mean composite (0-4)"); ax1.set_ylim(0, 4.1)
    ax1.set_title("Composite by judge x tutor\n(Anthropic tutors left, OpenAI right)", fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(True, axis="y", alpha=0.25)

    fams = sorted(set(cr["a"]) | set(cr["b"]))
    xf = range(len(fams))
    ax2.bar([i - w / 2 for i in xf], [100 * cr["a"][f] for f in fams], w, label=f"{ja} judge", color="#2563eb")
    ax2.bar([i + w / 2 for i in xf], [100 * cr["b"][f] for f in fams], w, label=f"{jb} judge", color="#dc2626")
    ax2.set_xticks(list(xf)); ax2.set_xticklabels(fams, fontsize=9)
    ax2.set_ylabel("% of dimension scores at ceiling (==4)"); ax2.set_ylim(0, 100)
    ax2.set_title("Ceiling rate by judge x tutor family\n(own-family inflation = blue tall on 'anthropic')", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    path = assets / "judge_bias.png"
    fig.savefig(path, dpi=130)
    print(f"  figure -> {path}")


if __name__ == "__main__":
    main()

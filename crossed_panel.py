"""Crossed-panel judge self-preference: is the Opus judge's own-family inflation real, or
just leniency toward strong tutors? Combines the frontier + crossed runs and controls for tier.

Runs src.validation.judge_bias.run_crossed_panel_analysis, prints the tables, and writes
docs/crossed_panel.json (+ docs/assets/crossed_panel.png). No new API calls (reads scores).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.validation.judge_bias import run_crossed_panel_analysis

A_DIRS = ["results/frontier/_scores/opus-4-8", "results/crossed/_scores/opus-4-8"]
B_DIRS = ["results/frontier/_scores/gpt-5.5", "results/crossed/_scores/gpt-5.5"]


def _f(v):
    return " n/a" if v is None else f"{v:+.3f}"


def main() -> None:
    res = run_crossed_panel_analysis(A_DIRS, B_DIRS)
    ja, jb = res["judges"]["a"], res["judges"]["b"]
    print(f"\nJudge A={ja}  Judge B={jb}  n={res['n_conversations']} conversations\n")

    print("Per-tutor mean composite (sorted family, tier):")
    pt = res["per_tutor"]
    order = sorted(pt, key=lambda t: (pt[t]["family"], pt[t]["tier"] != "strong", t))
    print(f"  {'tutor':<24} {'fam':<10} {'tier':<7} {'n':>3} {'A':>7} {'B':>7} {'A-B':>8}")
    for t in order:
        d = pt[t]
        print(f"  {t:<24} {d['family']:<10} {d['tier']:<7} {d['n']:>3} {d['a_mean']:7.3f} {d['b_mean']:7.3f} {d['delta']:+8.3f}")

    print("\n2x2 cell mean delta (A-B) by family/tier:")
    for cell, d in sorted(res["cell_delta_means"].items()):
        print(f"  {cell:<20} n={d['n']:>3}  mean delta={_f(d['mean'])}")

    print("\n--- Is the family effect real, or tier (quality) leniency? ---")
    print(f"  naive family DiD (uncontrolled)        = {_f(res['naive_family_did'])}")
    pc = res["per_tier_contrast"]
    for tier in ("strong", "weak"):
        if tier in pc:
            print(f"  within-{tier:<6} family contrast        = {_f(pc[tier])}")
    print(f"  TIER-CONTROLLED family effect          = {_f(res['tier_controlled_family_effect'])}")
    print(f"  stratified permutation p               = {res['stratified_permutation_p']:.4f}")

    print("\nPer-dimension family effect (does any dimension survive tier control?):")
    print(f"  {'dim':<5}{'naive':>9}{'within-strong':>15}{'tier-ctrl':>11}")
    for dim, d in res["per_dimension_tier_controlled"].items():
        print(f"  {dim:<5}{_f(d['naive']):>9}{_f(d['within_strong']):>15}{_f(d['tier_controlled']):>11}")

    _figure(res)
    out = Path("docs/crossed_panel.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"\n  results -> {out}")


def _figure(res: dict) -> None:
    pt = res["per_tutor"]
    order = sorted(pt, key=lambda t: (pt[t]["family"], pt[t]["tier"] != "strong", t))
    assets = Path("docs/assets"); assets.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.1, 1]})

    x = range(len(order)); w = 0.38
    ax1.bar([i - w / 2 for i in x], [pt[t]["a_mean"] for t in order], w, label=f"{res['judges']['a']} judge", color="#2563eb")
    ax1.bar([i + w / 2 for i in x], [pt[t]["b_mean"] for t in order], w, label=f"{res['judges']['b']} judge", color="#dc2626")
    labels = [f"{t.replace('claude-','')}\n({pt[t]['family'][:4]}/{pt[t]['tier'][:1]})" for t in order]
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("Mean composite (0-4)"); ax1.set_ylim(0, 4.1)
    ax1.set_title("Composite by judge x tutor (crossed panel)\nnew cells: gpt-5.5 (strong OpenAI), haiku-weak (weak Anthropic)", fontsize=9)
    ax1.legend(fontsize=8); ax1.grid(True, axis="y", alpha=0.25)

    pc = res["per_tier_contrast"]
    bars = [("naive\nDiD", res["naive_family_did"]),
            ("within\nstrong", pc.get("strong")),
            ("within\nweak", pc.get("weak")),
            ("tier-\ncontrolled", res["tier_controlled_family_effect"])]
    names = [b[0] for b in bars]; vals = [b[1] or 0 for b in bars]
    colors = ["#888", "#1f5673", "#1f5673", "#7a1f2b"]
    ax2.bar(range(len(vals)), vals, color=colors)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("Anthropic-favoring delta (pts)")
    ax2.set_title(f"Family effect, controlled for tier\nstratified p={res['stratified_permutation_p']:.3f}", fontsize=9)
    for i, v in enumerate(vals):
        ax2.text(i, v + (0.01 if v >= 0 else -0.03), f"{v:+.2f}", ha="center", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    path = assets / "crossed_panel.png"
    fig.savefig(path, dpi=130)
    print(f"  figure  -> {path}")


if __name__ == "__main__":
    main()

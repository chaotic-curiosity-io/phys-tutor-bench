"""Construct-validity / dimension-redundancy probe: are the six dimensions really six?

Runs src.validation.dimensionality on the scored conversations, prints the tables, and writes
docs/construct_probe.json (+ docs/assets/construct_validity.png). No new API calls.

Usage:
    python construct_probe.py [opus_dir] [gpt_dir] [all_results_dir]
Defaults to the frontier dual-judge run + the full 144-score panel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.validation.dimensionality import run_dimensionality_analysis

OPUS = sys.argv[1] if len(sys.argv) > 1 else "results/frontier/_scores/opus-4-8"
GPT = sys.argv[2] if len(sys.argv) > 2 else "results/frontier/_scores/gpt-5.5"
ALL = sys.argv[3] if len(sys.argv) > 3 else "results"


def _heat(ax, M, labels_r, labels_c, title):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels_c))); ax.set_xticklabels(labels_c, fontsize=8)
    ax.set_yticks(range(len(labels_r))); ax.set_yticklabels(labels_r, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i][j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title(title, fontsize=10)
    return im


def main() -> None:
    res = run_dimensionality_analysis(OPUS, GPT, ALL)
    wp, cf, halo = res["within_pooled"], res["cross_frontier"], res["halo"]
    labels = wp["labels"]

    print(f"\nn: within-pooled {res['n']['within_pooled']}, within-frontier {res['n']['within_frontier']}, "
          f"cross-frontier {res['n']['cross_frontier']}")

    print("\n=== Within-judge (pooled 144; halo-INFLATED) ===")
    print(f"  mean |inter-dimension r| = {wp['mean_offdiag']:.3f}")
    print(f"  Cronbach alpha           = {wp['cronbach_alpha']:.3f}")
    e = wp["eigen"]
    print(f"  PC1 explains {100*e['pc1_frac']:.1f}% of variance; Kaiser factors (eigenvalue>1) = {e['kaiser_factors']}")
    print(f"  eigenvalues: {[round(x,2) for x in e['eigenvalues']]}")
    print(f"  redundancy pairs |r|>=0.8: {wp['redundancy_pairs_0.8'] or 'none'}")

    print("\n=== Cross-judge MTMM (frontier 48; halo-FREE) ===")
    print(f"  mean convergent (same-dim, A vs B reliability) = {cf['mean_convergent']:.3f}")
    print(f"  mean heterotrait (diff-dim, halo-free)         = {cf['mean_heterotrait']:.3f}")
    print(f"  Campbell-Fiske violations = {cf['campbell_fiske_violations']}/{cf['campbell_fiske_comparisons']}")
    print("  per-dimension cross-judge reliability (convergent diagonal):")
    for d, v in cf["convergent_diagonal"].items():
        print(f"    {d:<4} {v:+.3f}")
    es = cf["eigen_symmetrized"]
    print(f"  symmetrized PC1 = {100*es['pc1_frac']:.1f}%; Kaiser factors = {es['kaiser_factors']}")

    print("\n=== Halo isolation (same 48 frontier conversations) ===")
    print(f"  within-judge mean |inter-dim r| = {halo['within_frontier_mean_offdiag']:.3f}")
    print(f"  cross-judge mean heterotrait    = {halo['cross_frontier_mean_heterotrait']:.3f}")
    print(f"  => halo inflation               = {halo['halo_inflation']:+.3f}")

    # ---- figure: within-judge corr (left) + cross-judge MTMM (right) ----
    assets = Path("docs/assets"); assets.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5))
    _heat(ax1, np.array(wp["matrix"]), labels, labels,
          f"Within-judge (pooled n={res['n']['within_pooled']})\nhalo-inflated; PC1={100*e['pc1_frac']:.0f}%")
    im = _heat(ax2, np.array(cf["mtmm_matrix"]), labels, labels,
               f"Cross-judge MTMM (frontier n={res['n']['cross_frontier']})\nrows=Opus dim, cols=GPT dim; diagonal=reliability")
    ax2.set_xlabel("GPT-5.5 judge", fontsize=8); ax2.set_ylabel("Opus judge", fontsize=8)
    fig.colorbar(im, ax=[ax1, ax2], fraction=0.025, pad=0.02, label="Pearson r")
    path = assets / "construct_validity.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\n  figure  -> {path}")

    out = Path("docs/construct_probe.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"  results -> {out}")


if __name__ == "__main__":
    main()

"""Judge self-preference (own-family bias): does an LLM judge over-reward its own family?

PhysTutorBench's frontier run scored every conversation with two independent judges —
Claude Opus 4.8 (Anthropic family) and GPT-5.5 (OpenAI family). If a judge favors tutors
from its own family, the two judges will disagree in a *family-aligned* way.

Design — paired difference-in-differences (DiD). Both judges scored the SAME conversations,
so for each conversation take the within-conversation delta d = score(judge A) - score(judge B);
conversation difficulty and overall judge leniency cancel. Self-preference predicts that d is
larger for judge A's own family than for the other family:

    DiD = mean(d | tutor family == A's family) - mean(d | tutor family == B's family)

DiD > 0 ==> judge A inflates its own family relative to judge B (jointly: both judges' own-family
pulls push the two family means apart). Significance via a label-permutation test; uncertainty via
a paired bootstrap.

The confound this CANNOT resolve: in the frontier panel, family is confounded with quality (all
Anthropic tutors are strong, the one OpenAI tutor is weak). A positive DiD is therefore consistent
with self-preference OR with a judge-specific leniency at the top of the scale. Separating them
needs a CROSSED panel (strong and weak tutors in both families). Reported honestly by the driver.
"""

from __future__ import annotations

import numpy as np

from src.scenarios.schema import ConversationScore
from src.scoring.rubric import ALL_DIMENSIONS

DIMENSION_IDS = [d.id for d in ALL_DIMENSIONS]
MAX_SCORE = 4


def model_family(name: str) -> str:
    """Map a model name to its provider family (mirrors create_tutor_backend's prefixes)."""
    if name.startswith("claude-"):
        return "anthropic"
    if name.startswith("gpt-") or name[:2] in ("o1", "o3", "o4", "o5"):
        return "openai"
    return "other"


# A-priori quality tier for the crossed-panel analysis: current models with a good tutoring
# prompt are "strong"; the answer-dumper variant and the older gpt-4o are "weak". The point of
# the crossed panel is to put both families in BOTH tiers so family is no longer confounded
# with quality. "weak" claude-haiku-4-5-weak is weak by PROMPT; gpt-4o is weak by CAPABILITY.
TIER_STRONG = {"claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5", "gpt-5.5"}
TIER_WEAK = {"gpt-4o", "claude-haiku-4-5-weak"}


def tier_of(model: str) -> str:
    if model in TIER_STRONG:
        return "strong"
    if model in TIER_WEAK:
        return "weak"
    return "unknown"


def _dims(cs: ConversationScore) -> dict[str, int]:
    return {ds.dimension: ds.score for ds in cs.scores}


def align_judges(
    a_scores: list[ConversationScore],
    b_scores: list[ConversationScore],
) -> list[dict]:
    """Inner-join two judges' scores on conversation_id (judge A = first arg, B = second).

    Each row carries the tutor, its family, both composites, and both per-dimension maps.
    """
    a_by = {cs.conversation_id: cs for cs in a_scores}
    b_by = {cs.conversation_id: cs for cs in b_scores}
    rows = []
    for cid in sorted(set(a_by) & set(b_by)):
        a, b = a_by[cid], b_by[cid]
        rows.append({
            "conversation_id": cid,
            "tutor": a.model_under_test,
            "family": model_family(a.model_under_test),
            "tier": tier_of(a.model_under_test),
            "a_comp": a.composite_score,
            "b_comp": b.composite_score,
            "a_dims": _dims(a),
            "b_dims": _dims(b),
        })
    return rows


def _delta(row: dict, dim: str | None) -> float:
    if dim is None:
        return row["a_comp"] - row["b_comp"]
    return row["a_dims"].get(dim, 0) - row["b_dims"].get(dim, 0)


def paired_deltas(rows: list[dict], dim: str | None = None) -> dict[str, list[float]]:
    """{family -> [within-conversation deltas (judge A - judge B)]}."""
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["family"], []).append(_delta(r, dim))
    return out


def difference_in_differences(rows: list[dict], family_a: str, family_b: str,
                              dim: str | None = None) -> float | None:
    """mean(delta | family_a) - mean(delta | family_b). None if either group is empty."""
    d = paired_deltas(rows, dim)
    if not d.get(family_a) or not d.get(family_b):
        return None
    return float(np.mean(d[family_a]) - np.mean(d[family_b]))


def permutation_test(rows: list[dict], family_a: str, family_b: str,
                     dim: str | None = None, n_perm: int = 10000, seed: int = 0) -> dict:
    """Two-sided label-permutation test for the DiD.

    Shuffles the family labels across conversations (preserving the group sizes) and recomputes
    the DiD to build the null distribution. p = fraction of permutations with |DiD| >= |observed|.
    """
    rows = [r for r in rows if r["family"] in (family_a, family_b)]
    observed = difference_in_differences(rows, family_a, family_b, dim)
    if observed is None:
        return {"observed": None, "p_value": None, "n_perm": 0, "null_mean": None}
    deltas = np.array([_delta(r, dim) for r in rows])
    labels = np.array([r["family"] for r in rows])
    n_a = int(np.sum(labels == family_a))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(deltas))
    count = 0
    null_vals = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(idx)
        grp_a = deltas[perm[:n_a]]
        grp_b = deltas[perm[n_a:]]
        diff = grp_a.mean() - grp_b.mean()
        null_vals[k] = diff
        if abs(diff) >= abs(observed) - 1e-12:
            count += 1
    return {"observed": float(observed), "p_value": count / n_perm,
            "n_perm": n_perm, "null_mean": float(null_vals.mean())}


def bootstrap_ci(rows: list[dict], family_a: str, family_b: str, dim: str | None = None,
                 n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the DiD, resampling conversations within each family group."""
    d = paired_deltas(rows, dim)
    if not d.get(family_a) or not d.get(family_b):
        return None
    a = np.array(d[family_a]); b = np.array(d[family_b])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        ba = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        boots[k] = ba.mean() - bb.mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def ceiling_rates(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Fraction of per-dimension scores at the maximum (==4), by judge ('a'/'b') and family."""
    counts: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        for judge, key in (("a", "a_dims"), ("b", "b_dims")):
            c = counts.setdefault((judge, r["family"]), [0, 0])
            for v in r[key].values():
                c[0] += int(v == MAX_SCORE)
                c[1] += 1
    out: dict[str, dict[str, float]] = {"a": {}, "b": {}}
    for (judge, fam), (n4, n) in counts.items():
        out[judge][fam] = n4 / n if n else None
    return out


def cell_means(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Mean composite per judge ('a'/'b') x tutor."""
    acc: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        acc.setdefault(("a", r["tutor"]), []).append(r["a_comp"])
        acc.setdefault(("b", r["tutor"]), []).append(r["b_comp"])
    out: dict[str, dict[str, float]] = {"a": {}, "b": {}}
    for (judge, tutor), vals in acc.items():
        out[judge][tutor] = float(np.mean(vals))
    return out


def per_dimension_did(rows: list[dict], family_a: str, family_b: str) -> dict[str, float | None]:
    return {dim: difference_in_differences(rows, family_a, family_b, dim) for dim in DIMENSION_IDS}


def _delta_comp(row: dict) -> float:
    return row["a_comp"] - row["b_comp"]


def cell_delta_means(rows: list[dict]) -> dict:
    """Mean composite delta (judge A - judge B) per (family, tier) cell of the 2x2."""
    acc: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        acc.setdefault((r["family"], r["tier"]), []).append(_delta_comp(r))
    return {f"{fam}/{tier}": {"mean": float(np.mean(v)), "n": len(v)} for (fam, tier), v in acc.items()}


def per_tutor_means(rows: list[dict]) -> dict:
    """Per-tutor judge-A mean, judge-B mean, delta, family, tier, n."""
    acc: dict[str, list[dict]] = {}
    for r in rows:
        acc.setdefault(r["tutor"], []).append(r)
    out = {}
    for tutor, rs in acc.items():
        out[tutor] = {
            "a_mean": float(np.mean([x["a_comp"] for x in rs])),
            "b_mean": float(np.mean([x["b_comp"] for x in rs])),
            "delta": float(np.mean([_delta_comp(x) for x in rs])),
            "family": rs[0]["family"], "tier": rs[0]["tier"], "n": len(rs),
        }
    return out


def tier_controlled_family_effect(rows: list[dict], family_a: str = "anthropic",
                                  family_b: str = "openai",
                                  tiers: tuple[str, ...] = ("strong", "weak")) -> dict:
    """Family effect on the A-B delta, holding tier constant.

    Within each tier, contrast = mean(delta | family_a) - mean(delta | family_b); the
    tier-controlled effect is the equal-weight average of the per-tier contrasts (so the
    larger tier doesn't dominate). If the naive (uncontrolled) family DiD is mostly quality
    leniency, this collapses toward zero; if it is genuine self-preference, it persists.
    """
    contrasts: dict[str, float] = {}
    for tier in tiers:
        a = [_delta_comp(r) for r in rows if r["family"] == family_a and r["tier"] == tier]
        b = [_delta_comp(r) for r in rows if r["family"] == family_b and r["tier"] == tier]
        if a and b:
            contrasts[tier] = float(np.mean(a) - np.mean(b))
    effect = float(np.mean(list(contrasts.values()))) if contrasts else None
    return {"per_tier_contrast": contrasts, "tier_controlled_effect": effect}


def stratified_permutation_test(rows: list[dict], family_a: str = "anthropic",
                                family_b: str = "openai",
                                tiers: tuple[str, ...] = ("strong", "weak"),
                                n_perm: int = 10000, seed: int = 0) -> dict:
    """Permutation null for the tier-controlled family effect: shuffle family labels WITHIN
    each tier (so the test isolates family from tier)."""
    rows = [r for r in rows if r["family"] in (family_a, family_b) and r["tier"] in tiers]
    obs = tier_controlled_family_effect(rows, family_a, family_b, tiers)["tier_controlled_effect"]
    if obs is None:
        return {"observed": None, "p_value": None, "n_perm": 0}
    rng = np.random.default_rng(seed)
    tdata = []
    for tier in tiers:
        rs = [r for r in rows if r["tier"] == tier]
        deltas = np.array([_delta_comp(r) for r in rs])
        na = int(np.sum([r["family"] == family_a for r in rs]))
        if 0 < na < len(deltas):
            tdata.append((deltas, na))
    count = 0
    for _ in range(n_perm):
        contrasts = []
        for deltas, na in tdata:
            perm = rng.permutation(len(deltas))
            contrasts.append(deltas[perm[:na]].mean() - deltas[perm[na:]].mean())
        if contrasts and abs(float(np.mean(contrasts))) >= abs(obs) - 1e-12:
            count += 1
    return {"observed": float(obs), "p_value": count / n_perm, "n_perm": n_perm}


def per_dimension_tier_controlled(rows: list[dict], family_a: str = "anthropic",
                                  family_b: str = "openai") -> dict:
    """For each dimension: naive family DiD, within-strong contrast, tier-controlled effect."""
    out = {}
    for d in ALL_DIMENSIONS:
        dr = [{**r, "a_comp": r["a_dims"].get(d.id, 0), "b_comp": r["b_dims"].get(d.id, 0)} for r in rows]
        tc = tier_controlled_family_effect(dr, family_a, family_b)
        out[d.abbreviation] = {
            "naive": difference_in_differences(dr, family_a, family_b),
            "within_strong": tc["per_tier_contrast"].get("strong"),
            "tier_controlled": tc["tier_controlled_effect"],
        }
    return out


def run_crossed_panel_analysis(a_dirs: list, b_dirs: list,
                               a_name: str = "claude-opus-4-8", b_name: str = "gpt-5.5") -> dict:
    """Crossed 2x2: combine frontier + crossed scores and de-confound family from tier.

    a_dirs / b_dirs are lists of score directories for judge A and judge B respectively.
    """
    from src.scoring.scorecard import load_scores

    a_scores = [s for d in a_dirs for s in load_scores(d)]
    b_scores = [s for d in b_dirs for s in load_scores(d)]
    rows = align_judges(a_scores, b_scores)
    fam_a, fam_b = model_family(a_name), model_family(b_name)

    naive = difference_in_differences(rows, fam_a, fam_b)
    tc = tier_controlled_family_effect(rows, fam_a, fam_b)
    perm = stratified_permutation_test(rows, fam_a, fam_b)
    return {
        "judges": {"a": a_name, "a_family": fam_a, "b": b_name, "b_family": fam_b},
        "n_conversations": len(rows),
        "naive_family_did": naive,
        "tier_controlled_family_effect": tc["tier_controlled_effect"],
        "per_tier_contrast": tc["per_tier_contrast"],
        "stratified_permutation_p": perm["p_value"],
        "cell_delta_means": cell_delta_means(rows),
        "per_tutor": per_tutor_means(rows),
        "per_dimension_tier_controlled": per_dimension_tier_controlled(rows),
    }


def run_judge_bias_analysis(a_dir, b_dir, a_name: str = "claude-opus-4-8",
                            b_name: str = "gpt-5.5") -> dict:
    """Full analysis. Judge A = a_dir's judge, Judge B = b_dir's judge."""
    from src.scoring.scorecard import load_scores

    rows = align_judges(load_scores(a_dir), load_scores(b_dir))
    fam_a = model_family(a_name)
    fam_b = model_family(b_name)

    composite_dd = difference_in_differences(rows, fam_a, fam_b)
    perm = permutation_test(rows, fam_a, fam_b)
    ci = bootstrap_ci(rows, fam_a, fam_b)
    deltas = paired_deltas(rows)

    return {
        "judges": {"a": a_name, "a_family": fam_a, "b": b_name, "b_family": fam_b},
        "n_conversations": len(rows),
        "composite_did": composite_dd,
        "did_permutation_p": perm["p_value"],
        "did_bootstrap_ci95": list(ci) if ci else None,
        "paired_delta_by_family": {
            f: {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v))}
            for f, v in deltas.items()
        },
        "per_dimension_did": per_dimension_did(rows, fam_a, fam_b),
        "ceiling_rates": ceiling_rates(rows),
        "cell_means": cell_means(rows),
    }

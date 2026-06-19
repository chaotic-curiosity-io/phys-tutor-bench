"""Build figures + analysis data for the lab report from the scored results."""

from __future__ import annotations

import json
from pathlib import Path

from src.scoring.scorecard import load_scores, Scorecard
from src.scoring.rubric import ALL_DIMENSIONS
from src.validation.construct_validity import run_construct_validity

S = load_scores("results")
print("scores:", len(S))
sc = Scorecard(S)

assets = Path("docs/assets")
assets.mkdir(parents=True, exist_ok=True)
sc.generate_comparison_plot(assets / "comparison.png")
sc.generate_heatmap(assets / "heatmap.png")
dist = sc.generate_distribution_plots(assets / "distributions")
print("distribution plots:", len(dist))

summary = sc.to_json()
per_topic = {m: sc.per_topic_breakdown(m) for m in sc.models}

out = {
    "models": sc.models,
    "dimensions": [{"id": d.id, "abbr": d.abbreviation, "name": d.name} for d in ALL_DIMENSIONS],
    "summary": summary,
    "per_topic": per_topic,
}
Path("docs/analysis_data.json").write_text(json.dumps(out, indent=2))
print("wrote docs/analysis_data.json")

# Construct validity (no API; correlations among dimensions across all 48 scores)
cv = run_construct_validity(S)
Path("docs/construct_validity.json").write_text(json.dumps(cv, indent=2, default=str))
print("wrote docs/construct_validity.json")

# Echo key tables for the report author
print("\n=== PER-TOPIC COMPOSITE (mean) ===")
for m in sc.models:
    row = {t: round(v["mean"], 2) for t, v in per_topic[m].items()}
    print(f"{m:14s} {row}")

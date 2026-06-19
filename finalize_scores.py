"""Turn raw judge scores into schema-valid ConversationScore files.

The Claude-as-judge step writes one raw JSON per conversation under
<run_dir>/raw_scores/<conversation_id>.json with shape:

    {
      "conversation_id": "...",
      "scenario_id": "...",
      "model_under_test": "...",
      "judge_model": "...",            # optional
      "dimensions": {
        "misconception_diagnosis":      {"score": 0-4, "justification": "..."},
        "scaffolding_strategy":         {"score": 0-4, "justification": "..."},
        "answer_disclosure_restraint":  {"score": 0-4, "justification": "..."},
        "conceptual_model_building":    {"score": 0-4, "justification": "..."},
        "transfer_success":             {"score": 0-4, "justification": "..."},
        "pedagogical_harm_avoidance":   {"score": 0-4, "justification": "..."}
      }
    }

This script computes the weighted composite with the project's rubric weights and
writes <run_dir>/scores/<conversation_id>_score.json so the scorecard/validation
tooling works unchanged. Centralizing the composite here keeps it consistent and
lets us re-weight without re-judging.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.scoring.rubric import compute_composite_score, ALL_DIMENSIONS, DEFAULT_WEIGHTS
from src.scenarios.schema import ConversationScore, DimensionScore
from src.scoring.scorecard import save_score

DIM_IDS = [d.id for d in ALL_DIMENSIONS]


def finalize(root: str) -> None:
    root_path = Path(root)
    raw_files = sorted(root_path.rglob("raw_scores/*.json"))
    print(f"found {len(raw_files)} raw score file(s) under {root_path}")

    ok = 0
    bad: list[tuple[str, str]] = []
    for rf in raw_files:
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
            dims = data["dimensions"]
            score_map: dict[str, int] = {}
            ds_list: list[DimensionScore] = []
            for did in DIM_IDS:
                entry = dims[did]
                s = int(entry["score"])
                if not (0 <= s <= 4):
                    raise ValueError(f"{did} score {s} out of range")
                score_map[did] = s
                ds_list.append(
                    DimensionScore(dimension=did, score=s, justification=str(entry["justification"]))
                )
            composite = compute_composite_score(score_map, DEFAULT_WEIGHTS)
            cs = ConversationScore(
                conversation_id=data["conversation_id"],
                scenario_id=data["scenario_id"],
                model_under_test=data["model_under_test"],
                judge_model=data.get("judge_model", "claude-code-judge"),
                scores=ds_list,
                composite_score=composite,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            save_score(cs, rf.parent.parent / "scores")
            ok += 1
        except Exception as e:  # noqa: BLE001
            bad.append((str(rf), str(e)))

    print(f"finalized {ok}; errors {len(bad)}")
    for path, err in bad:
        print(f"  BAD {path}: {err}")


if __name__ == "__main__":
    finalize(sys.argv[1] if len(sys.argv) > 1 else "results")

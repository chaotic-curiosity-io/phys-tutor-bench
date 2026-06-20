"""Estimate the API cost of the frontier run from saved transcripts (no new calls).

Total per-call tokens are exact (provider-reported, saved on each message). The
input/output split is reconstructed by re-tokenizing the generated message text
at ~4 chars/token, then input = (exact billed total) - (estimated output). Judge
calls were not metered, so their tokens are reconstructed from the exact judge
prompt (system + transcript) and the score JSON; GPT-5.5's hidden reasoning
tokens are NOT visible, so its judge cost is a LOWER BOUND.

v1 is omitted: local Ollama tutors + a local Ollama student (zero API tokens),
judged on a Claude Code subscription (flat-rate) ~= $0 marginal API cost.

Run from repo root:  python estimate_cost.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.engine.batch_runner import load_scenarios
from src.engine.conversation_loop import load_conversation
from src.scoring.llm_judge import JUDGE_SYSTEM_PROMPT, _format_conversation_for_judge
from src.scoring.rubric import format_rubric_for_prompt

# List prices, USD per 1M tokens (input, output). Anthropic: platform docs (cached
# 2026-06). OpenAI: gpt-5.5 $5/$30 and gpt-4o legacy $2.50/$10 (June 2026).
PRICES = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-5.5": (5.0, 30.0),
}
FRONTIER = Path("results/frontier")


def toks(s: str) -> int:
    return max(1, round(len(s) / 4))


def price(model: str, inp: float, out: float) -> float:
    pi, po = PRICES[model]
    return inp / 1e6 * pi + out / 1e6 * po


def main() -> None:
    scen = {s.id: s for s in load_scenarios("data/scenarios")}
    judge_sys = JUDGE_SYSTEM_PROMPT.format(rubric_text=format_rubric_for_prompt())

    tutor = defaultdict(lambda: {"in": 0, "out": 0, "n": 0})
    student = {"in": 0, "out": 0}
    student_model = None

    conv_paths = sorted(FRONTIER.rglob("*/conversations/*.json"))
    convs = {}  # id -> ConversationRecord
    for cp in conv_paths:
        rec = load_conversation(cp)
        convs[rec.id] = rec
        m = rec.model_under_test
        # combined (exact) per role; output estimated from message text.
        t_out = sum(toks(msg.content) for msg in rec.messages if msg.role == "tutor")
        s_out = sum(toks(msg.content) for msg in rec.messages if msg.role == "student")
        tutor[m]["out"] += t_out
        tutor[m]["in"] += max(0, rec.total_tutor_tokens - t_out)
        tutor[m]["n"] += 1
        student["out"] += s_out
        student["in"] += max(0, rec.total_student_tokens - s_out)
        student_model = rec.student_simulator_model

    # Judges: input = exact prompt (system + formatted transcript); output = score JSON.
    judges = defaultdict(lambda: {"in": 0, "out": 0, "n": 0})
    for sdir in sorted((FRONTIER / "_scores").glob("*")):
        if not sdir.is_dir():
            continue
        for sf in sdir.glob("*_score.json"):
            sd = json.loads(sf.read_text(encoding="utf-8"))
            rec = convs.get(sd["conversation_id"])
            if rec is None:
                continue
            jm = sd.get("judge_model", "")
            jkey = "gpt-5.5" if "gpt-5" in jm.lower() else "claude-opus-4-8"
            user = _format_conversation_for_judge(rec, scen[rec.scenario_id])
            judges[jkey]["in"] += toks(judge_sys) + toks(user)
            judges[jkey]["out"] += toks(json.dumps(sd.get("scores", [])))
            judges[jkey]["n"] += 1

    # ---- report ----
    print("FRONTIER RUN — estimated API cost (list prices; v1 ~= $0, local + subscription)\n")
    print(f"{'phase / model':<26}{'in_tok':>11}{'out_tok':>11}{'$ (est)':>10}")
    print("-" * 58)
    grand = 0.0
    rows = {}

    print("Conversations — tutor (model under test):")
    for m, d in sorted(tutor.items()):
        c = price(m, d["in"], d["out"]); grand += c; rows[f"tutor:{m}"] = c
        print(f"  {m:<24}{d['in']:>11,}{d['out']:>11,}{c:>10.2f}")

    sc = price(student_model, student["in"], student["out"]); grand += sc
    rows[f"student:{student_model}"] = sc
    print(f"Conversations — student (fixed):\n  {student_model:<24}{student['in']:>11,}{student['out']:>11,}{sc:>10.2f}")

    print("Judging (48 conversations each):")
    for jk, d in sorted(judges.items()):
        c = price(jk, d["in"], d["out"]); grand += c; rows[f"judge:{jk}"] = c
        note = "  (lower bound; reasoning tokens not metered)" if jk == "gpt-5.5" else ""
        print(f"  {jk:<24}{d['in']:>11,}{d['out']:>11,}{c:>10.2f}{note}")

    print("-" * 58)
    print(f"{'TOTAL (frontier run)':<48}{grand:>10.2f}")
    print(f"\nNote: total per-call tokens are exact; input/output split estimated (~4 chars/tok).")

    out = {"prices_per_1m": PRICES, "rows_usd": rows, "total_usd": round(grand, 2),
           "tutor": dict(tutor), "student": {"model": student_model, **student},
           "judges": {k: dict(v) for k, v in judges.items()}}
    (FRONTIER / "cost_estimate.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {FRONTIER / 'cost_estimate.json'}")


if __name__ == "__main__":
    main()

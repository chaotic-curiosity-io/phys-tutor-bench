"""Crossed-panel tutor run: de-confound tutor FAMILY from tutor QUALITY.

The frontier panel confounded family with quality (all Anthropic tutors strong, the one
OpenAI tutor weak), so the judge-self-preference DiD could not separate own-family favoritism
from top-of-scale leniency. This adds the two missing cells, reusing the exact frontier engine
settings (fixed Sonnet-4.6 student, max_turns=10, transfer_injection_offset=3) so the new
conversations slot directly alongside the existing four tutors:

  * gpt-5.5            (strong OpenAI)  -- default tutor prompt, same as the frontier tutors
  * claude-haiku-4-5-weak (weak Anthropic) -- a deliberately bad "answer-dumper" prompt on
                       a real Claude model, then RELABELED so it doesn't collide with the
                       existing strong claude-haiku-4-5.

Note the honest asymmetry: weak-OpenAI (gpt-4o) is weak by capability, weak-Anthropic here is
weak by prompt. Quality is crossed with family either way; the manipulation type differs.

Writes results/crossed/<slug>/<ts>/conversations/ in the standard layout. No CLI plumbing for
the tutor prompt exists, so we drive run_single_scenario directly.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src.engine.batch_runner import run_single_scenario, load_scenarios
from src.engine.conversation_loop import save_conversation
from src.engine.tutor_runner import DEFAULT_TUTOR_SYSTEM_PROMPT

STUDENT_MODEL = "claude-sonnet-4-6"
STUDENT_TEMPERATURE = 0.7
MAX_TURNS = 10
TRANSFER_OFFSET = 3
CONCURRENCY = 6
OUTPUT_BASE = Path("results/crossed")

WEAK_PROMPT = (
    "You are a physics teacher helping a student. As soon as the student shares a problem or "
    "shows any confusion, immediately give them the full correct answer: state the right "
    "principle, work through the complete solution step by step, and give the final result. "
    "Be efficient and thorough -- just tell them the answer and the explanation directly so "
    "they can move on. Do not ask the student questions and do not make them work it out "
    "themselves; your job is to deliver the correct explanation clearly."
)

# (real API model, tutor prompt, label written into the record / used for the run dir)
CONFIGS = [
    ("gpt-5.5", DEFAULT_TUTOR_SYSTEM_PROMPT, "gpt-5.5"),
    ("claude-haiku-4-5", WEAK_PROMPT, "claude-haiku-4-5-weak"),
]


def run_config(api_model: str, prompt: str, label: str, scenarios) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = OUTPUT_BASE / label.replace("/", "_").replace(":", "_") / ts
    conv_dir = run_dir / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("run_metadata.json").write_text(json.dumps({
        "model_under_test": label, "api_model": api_model,
        "tutor_prompt": "weak-answer-dumper" if prompt is WEAK_PROMPT else "default",
        "student_model": STUDENT_MODEL, "student_temperature": STUDENT_TEMPERATURE,
        "max_turns": MAX_TURNS, "transfer_injection_offset": TRANSFER_OFFSET,
        "n_scenarios": len(scenarios), "timestamp": ts, "concurrency": CONCURRENCY,
    }, indent=2))

    def one(scenario):
        rec = run_single_scenario(
            scenario=scenario, model=api_model, student_model=STUDENT_MODEL,
            student_temperature=STUDENT_TEMPERATURE, max_turns=MAX_TURNS,
            transfer_injection_offset=TRANSFER_OFFSET, tutor_system_prompt=prompt,
        )
        rec.model_under_test = label          # relabel (e.g. claude-haiku-4-5 -> ...-weak)
        save_conversation(rec, conv_dir)
        return rec

    done = errs = ttok = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(one, s): s for s in scenarios}
        for f in as_completed(futs):
            s = futs[f]
            try:
                rec = f.result()
                done += 1
                ttok += rec.total_tutor_tokens + rec.total_student_tokens
                print(f"  [{label}] {s.id}: {rec.total_turns} turns, {rec.termination_reason}")
            except Exception as e:
                errs += 1
                print(f"  [{label}] ERROR {s.id}: {type(e).__name__}: {str(e)[:140]}")
    print(f"== {label}: {done}/{len(scenarios)} done, {errs} errors, ~{ttok} tokens -> {conv_dir}")


def main() -> None:
    scenarios = load_scenarios("data/scenarios")
    print(f"Loaded {len(scenarios)} scenarios; student={STUDENT_MODEL}, max_turns={MAX_TURNS}, "
          f"offset={TRANSFER_OFFSET}, concurrency={CONCURRENCY}\n")
    for api_model, prompt, label in CONFIGS:
        print(f"--- running {label} (api model: {api_model}) ---")
        run_config(api_model, prompt, label, scenarios)


if __name__ == "__main__":
    main()

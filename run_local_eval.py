"""Run PhysTutorBench conversations with BOTH tutor and student served locally by Ollama.

The stock CLI only routes the *tutor* to a local OpenAI-compatible endpoint; the student
simulator stays on Anthropic. For a fully-local run (no API key) we drive the engine
directly here, pointing both the tutor (model under test) and the fixed student simulator
at the local Ollama server. Conversations are written in the same layout the scorecard
expects: results/<model_slug>/<timestamp>/conversations/<id>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src.engine.conversation_loop import ConversationLoop, save_conversation
from src.engine.student_simulator import StudentSimulator
from src.engine.tutor_runner import GenericHTTPTutor
from src.engine.batch_runner import load_scenarios


def run_one(scenario, tutor_model, student_model, ollama_base, max_turns):
    tutor = GenericHTTPTutor(base_url=ollama_base, model=tutor_model, api_key="ollama")
    student = StudentSimulator(
        scenario=scenario,
        model=student_model,
        api_base=ollama_base,
        api_key="ollama",
    )
    loop = ConversationLoop(
        scenario=scenario,
        tutor=tutor,
        student=student,
        max_turns=max_turns,
        transfer_injection_offset=3,
    )
    return loop.run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama tutor model under test (e.g. llama3.1:8b)")
    ap.add_argument("--student-model", default="qwen2.5:7b", help="Fixed local student simulator model")
    ap.add_argument("--scenarios-dir", default="data/scenarios")
    ap.add_argument("--ollama-base", default="http://localhost:11434")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--scenario-ids", default=None, help="Comma-separated subset of scenario IDs")
    ap.add_argument("--output", default="results")
    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios_dir)
    if args.scenario_ids:
        ids = {x.strip() for x in args.scenario_ids.split(",")}
        scenarios = [s for s in scenarios if s.id in ids]
    if not scenarios:
        print("No scenarios found.")
        sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    slug = args.model.replace("/", "_").replace(":", "_")
    run_dir = Path(args.output) / slug / ts
    conv_dir = run_dir / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "model_under_test": args.model,
        "student_model": args.student_model,
        "max_turns": args.max_turns,
        "n_scenarios": len(scenarios),
        "timestamp": ts,
        "ollama_base": args.ollama_base,
    }, indent=2))

    records, errors = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        fut = {
            ex.submit(run_one, s, args.model, args.student_model, args.ollama_base, args.max_turns): s
            for s in scenarios
        }
        for f in as_completed(fut):
            s = fut[f]
            try:
                rec = f.result()
                save_conversation(rec, conv_dir)
                records.append(rec)
                print(f"OK  {s.id}  turns={rec.total_turns}  term={rec.termination_reason}  "
                      f"tutor_tok={rec.total_tutor_tokens}", flush=True)
            except Exception as e:
                errors.append({"scenario_id": s.id, "error": str(e)})
                print(f"ERR {s.id}: {e}", flush=True)

    elapsed = round(time.time() - t0, 1)
    (run_dir / "run_summary.json").write_text(json.dumps({
        "total": len(scenarios),
        "completed": len(records),
        "errors": len(errors),
        "error_details": errors,
        "elapsed_sec": elapsed,
    }, indent=2))
    print(f"RUN_DIR {run_dir}")
    print(f"done: {len(records)}/{len(scenarios)} completed in {elapsed}s")


if __name__ == "__main__":
    main()

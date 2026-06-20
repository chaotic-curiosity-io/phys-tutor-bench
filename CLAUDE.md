# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PhysTutorBench evaluates how well an LLM *tutors* introductory physics (not whether it can solve it). It drives a model-under-test through simulated tutoring conversations where a student exhibits a misconception drawn from validated Physics Education Research (PER) instruments, then an LLM judge scores the tutor on six 0–4 pedagogical dimensions. A validation layer checks whether those scores agree with humans and discriminate known-quality tutors.

## Commands

```bash
pip install -e ".[dev]"          # install package + test deps (run from repo root)

pytest                            # full suite (testpaths=tests, asyncio_mode=auto)
pytest tests/test_scoring.py      # one file
pytest tests/test_scoring.py::TestRubric::test_composite_score -v   # one test

# Pipeline (each stage feeds the next; `phystutor` entry point = src.cli:cli)
phystutor generate-scenarios --topic mechanics --count 3   # taxonomy -> data/scenarios/<topic>/*.json
phystutor run-single --scenario <id-or-path> --model claude-sonnet-4-20250514   # debug one conversation
phystutor run-benchmark --model <model> --concurrency 5     # -> results/<model_slug>/<timestamp>/conversations/
phystutor score --results-dir results/<model_slug>/<timestamp>/   # -> .../scores/*_score.json
phystutor scorecard --results-dir results/ --compare modelA,modelB   # tables + PNGs + scorecard.json
phystutor annotate --conversations <dir>    # Streamlit human-annotation UI -> SQLite
phystutor validate --annotations <db> --results-dir <dir> [--run-discriminant]
```

No linter/formatter is configured (despite `.ruff_cache`/`.mypy_cache` in `.gitignore`). There is no async test despite `asyncio_mode=auto`.

## API keys

`generate-scenarios`, `run-benchmark`, `score`, and `validate` make real API calls and need `ANTHROPIC_API_KEY` (and `OPENAI_API_KEY` to evaluate GPT models as the tutor). Most unit tests inject `MagicMock` clients and need no network — **every API-calling class takes a `client=` parameter, which is the seam for testing** (`StudentSimulator`, `AnthropicTutor`, `OpenAITutor`, `LLMJudge`). A few tests that construct a real backend still require the key env var to be *present* (any value) even though they make no call.

## Architecture: a four-stage file-based pipeline

Stages communicate through JSON files on disk, not in-memory objects. `src/scenarios/schema.py` defines the Pydantic models that are the contract between every stage — read it first.

```
taxonomy ──generate──> Scenario JSON ──run──> ConversationRecord JSON ──score──> ConversationScore JSON ──> Scorecard
(data/taxonomies/)     (data/scenarios/)       (results/.../conversations/)       (results/.../scores/)
```

1. **Scenarios** (`src/scenarios/`): `taxonomy.py` loads the 65+ misconception taxonomy (the "backbone" — every scenario traces to a PER instrument). `generate_scenarios.py` prompts an LLM to author `Scenario` objects per misconception, allocating counts to hit per-topic targets (`TOPIC_TARGETS`).
2. **Engine** (`src/engine/`): `conversation_loop.py` runs one tutoring dialogue. `student_simulator.py` is a **controlled variable** — fixed model/prompt/temperature; changing it invalidates prior results. `tutor_runner.py` is the model-under-test, selected by `create_tutor_backend()` (a factory dispatching on model-name prefix). `batch_runner.py` fans scenarios across a `ThreadPoolExecutor` with a pre-flight cost estimate/abort.
3. **Scoring** (`src/scoring/`): `rubric.py` holds the six dimensions as structured data + composite weights. `judge.py` is the `Judge` ABC; `llm_judge.py` is the working implementation (frontier model, temp 0, JSON output); `reward_model_judge.py` is an intentional stub for a future trained model. `scorecard.py` aggregates and plots.
4. **Validation** (`src/validation/`): `agreement_analysis.py` (Cohen's/weighted kappa, Krippendorff's alpha vs. human SQLite annotations), `construct_validity.py`, and `discriminant_validity.py` (generates gold/foil tutor conversations and asserts the rubric ranks `expert > pseudo_socratic > answer_dumper > wrong_physics`). `human_annotation_interface.py` is the Streamlit app.

### The six dimensions (defined once in `rubric.py`)

MD (misconception diagnosis), SS (scaffolding), ADR (answer-disclosure restraint), CMB (conceptual model building), TS (transfer success), PHA (pedagogical harm avoidance). All are 0–4 with justifications; composite is a weighted mean. Add/rename a dimension here and the judge prompt, scorecard, and validation pick it up automatically.

## Conventions and gotchas that span files

- **Run from the repo root.** The package is named `src` and everything imports as `from src.…`; there's no installed package alias.
- **Config is mostly not wired in.** The CLI reads only `config.student_simulator`. The `judge`, `conversation` (incl. `tutor_system_prompt`, `transfer_injection_offset`), `batch`, and `scoring_weights` sections of `config/default.yaml` are **not** consumed — those defaults live as Click option defaults and module constants (`DEFAULT_TUTOR_SYSTEM_PROMPT`, `rubric.DEFAULT_WEIGHTS`). Editing the YAML's weights or prompts will not change behavior until you thread them through.
- **Filenames are the contract between stages.** Scores are written as `{conversation_id}_score.json`; `score` treats every other `*.json` (except `run_metadata.json`/`run_summary.json`) as a conversation, and `load_scores` globs `*_score.json`. Don't rename these.
- **Scenario IDs encode topic.** `_make_scenario_id` produces `mech-/em-/thwv-/modq-` prefixes, and `Scorecard._infer_topic` parses that prefix for per-topic breakdowns. Keep the prefix scheme consistent.
- **Conversation loop keeps two role-swapped histories.** From the tutor's view student=`user`/tutor=`assistant`; from the student's view it's reversed. The tutor always gets the last word at `max_turns`.
- **Transfer is the outcome measure.** The student simulator is told to inject the `transfer_problem` `transfer_injection_offset` turns before the end (via an appended system-prompt instruction), and the student answers honestly from its *current* understanding — that exchange drives the TS score.
- **LLM JSON is regex-extracted.** Both the judge and the scenario generator pull the first `{…}` block out of the response, so prompt edits that change output shape can silently break parsing.
- `results/` and `*.db` are git-ignored (runtime artifacts); `data/scenarios/` is generated and not committed.
```

## Published papers & docs (GitHub Pages)

`docs/` is served as a static site by **GitHub Pages — source: `main` branch, `/docs` path** → https://chaotic-curiosity-io.github.io/phys-tutor-bench/. The research write-ups live there as styled, self-contained HTML:
- `index.html` — v1 empirical report
- `methodology.html` — v2 PER-grounded methodology (§6 = the ten future research directions)
- `v3-plan.html` — v3 validity-bridge implementation plan
- `concept-paper.html` — fundable concept paper / prospectus for PER groups & funders
- `research_corpus.json` — primary-source-verified evidence base behind the papers

**Publishing a new paper (standing rule):** add it to `docs/` as styled HTML (model new ones on `v3-plan.html` / `concept-paper.html` — shared CSS, nav backlinks top and bottom, `og:`/`twitter:` link-preview meta), cross-link it into the other papers' nav **and** the root `README.md`, then **commit to `main`** — Pages auto-builds; a feature branch would NOT deploy. Verify the live URL after the build settles.

**Keep READMEs in sync (standing rule):** whenever docs, commands, structure, or behavior change, update `README.md` (and any other READMEs) in the *same* change — never defer it to "later."

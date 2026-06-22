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
phystutor run-single --scenario <id-or-path> --model claude-opus-4-8   # debug one conversation
phystutor run-benchmark --model <model> [--student-model M] [--transfer-injection-offset N] --concurrency 5  # -> results/<slug>/<ts>/conversations/
phystutor score --results-dir <run-or-base> --judge-model <model> [--output <dir>]   # -> <dir>/*_score.json (per-judge dir for dual-judge runs)
phystutor scorecard --results-dir results/ --compare modelA,modelB   # tables + PNGs + scorecard.json
python compare_judges.py --a <scoresA> --b <scoresB>   # inter-judge kappa/alpha across two judges' score dirs
phystutor annotate --conversations <dir>    # Streamlit human-annotation UI -> SQLite
phystutor validate --annotations <db> --results-dir <dir> [--run-discriminant]

# Fully-local + manual-judge reproduction path (no CLI; used for the v1 Ollama report)
python run_local_eval.py        # drives BOTH tutor AND student via local Ollama -> results/<slug>/<ts>/conversations/
#   ...judge each conversation with judge_rubric.md -> <run>/raw_scores/<id>.json  (score+justification, NO composite)
python finalize_scores.py <run> # apply rubric weights -> <run>/scores/<id>_score.json  (re-weight without re-judging)
python compute_agreement.py results raw_scores raw_scores_b  # judge test-retest self-consistency (κ/α), NOT inter-judge
python docs_build_data.py       # rebuild docs/ figures + analysis_data.json from scored results
python bridge_probe.py          # in-silico criterion-validity probe: process composite -> transfer (no API) -> docs/bridge_probe.json + assets/bridge_scatter.png
python judge_bias.py            # judge self-preference (own-family bias): paired DiD on the dual-judge run (no API) -> docs/judge_bias.json + assets/judge_bias.png
python construct_probe.py       # construct validity / dimension redundancy: within-judge vs cross-judge MTMM (no API) -> docs/construct_probe.json + assets/construct_validity.png
python run_crossed.py           # PAID: adds gpt-5.5 + a weak-PROMPT haiku tutor -> results/crossed/ (relabels haiku as ...-weak; then `score` with both judges)
python crossed_panel.py         # tier-controlled 2x2 self-preference (frontier+crossed scores, no API) -> docs/crossed_panel.json + assets/crossed_panel.png
```

No linter/formatter is configured (despite `.ruff_cache`/`.mypy_cache` in `.gitignore`). There is no async test despite `asyncio_mode=auto`.

## API keys

`generate-scenarios`, `run-benchmark`, `score`, and `validate` make real API calls and need `ANTHROPIC_API_KEY` (and `OPENAI_API_KEY` to use a GPT model as the **tutor or judge**). Most unit tests inject `MagicMock` clients and need no network — **every API-calling class takes a `client=` parameter, which is the seam for testing** (`StudentSimulator`, `AnthropicTutor`, `OpenAITutor`, `LLMJudge`, `OpenAIJudge`). A few tests that construct a real backend still require the key env var to be *present* (any value) even though they make no call.

**Newer frontier models reject `temperature`** (Opus 4.8/4.7, Fable/Mythos 5; OpenAI o-series & GPT-5.x). `src/engine/model_compat.py` builds per-model request kwargs (drops `temperature`, uses `max_completion_tokens` for OpenAI reasoning models) and is the single place all four backends go through — so any role can use any of these models without a 400. Models like `claude-sonnet-4-6`, `claude-haiku-4-5`, and `gpt-4o` still take `temperature`.

## Architecture: a four-stage file-based pipeline

Stages communicate through JSON files on disk, not in-memory objects. `src/scenarios/schema.py` defines the Pydantic models that are the contract between every stage — read it first.

```
taxonomy ──generate──> Scenario JSON ──run──> ConversationRecord JSON ──score──> ConversationScore JSON ──> Scorecard
(data/taxonomies/)     (data/scenarios/)       (results/.../conversations/)       (results/.../scores/)
```

1. **Scenarios** (`src/scenarios/`): `taxonomy.py` loads the 65+ misconception taxonomy (the "backbone" — every scenario traces to a PER instrument). `generate_scenarios.py` prompts an LLM to author `Scenario` objects per misconception, allocating counts to hit per-topic targets (`TOPIC_TARGETS`).
2. **Engine** (`src/engine/`): `conversation_loop.py` runs one tutoring dialogue. `student_simulator.py` is a **controlled variable** — fixed model/prompt/temperature; changing it invalidates prior results. `tutor_runner.py` is the model-under-test, selected by `create_tutor_backend()` (a factory dispatching on model-name prefix — `AnthropicTutor`/`OpenAITutor`, **or `GenericHTTPTutor` when an explicit `api_base` is passed, which takes precedence over prefix matching** so a locally-served model like `gpt-oss` reaches the local endpoint, not the cloud; `--api-base`/`--api-key` expose this on `run-single`/`run-benchmark`). `batch_runner.py` fans scenarios across a `ThreadPoolExecutor` with a pre-flight cost estimate/abort.
3. **Scoring** (`src/scoring/`): `rubric.py` holds the six dimensions as structured data + composite weights. `judge.py` is the `Judge` ABC **plus the `create_judge(model)` factory** (dispatches on model prefix, mirroring `create_tutor_backend`); `llm_judge.py` is the Anthropic implementation and `openai_judge.py` is the OpenAI one (both reuse the same rubric prompt + JSON parsing); `reward_model_judge.py` is an intentional stub for a future trained model. `scorecard.py` aggregates and plots. **Two scoring tracks exist:** the in-process `phystutor score` (judge returns a fully-composited `ConversationScore`), and a file-based manual-judge track where a judge follows root `judge_rubric.md` to write *raw* dimension scores to `<run>/raw_scores/<id>.json` (no composite) and `finalize_scores.py` applies `rubric.DEFAULT_WEIGHTS` to emit `<run>/scores/<id>_score.json` — so you can re-weight without re-judging. Root `compare_judges.py` computes **inter-judge** agreement (two judge models); `compute_agreement.py` computes **test–retest self-consistency** (one judge, two passes) — both reuse `validation/agreement_analysis.py`.
4. **Validation** (`src/validation/`): `agreement_analysis.py` (Cohen's/weighted kappa, Krippendorff's alpha vs. human SQLite annotations), `construct_validity.py`, `discriminant_validity.py` (generates gold/foil tutor conversations and asserts the rubric ranks `expert > pseudo_socratic > answer_dumper > wrong_physics`), and `predictive_validity.py` (criterion validity: does the process composite predict the transfer-success outcome? regresses one judge's process against the **other** judge's TS to break shared-method circularity; driven by root `bridge_probe.py`), and `judge_bias.py` (judge self-preference / own-family bias via a paired difference-in-differences on the dual-judge run, plus a tier-controlled crossed-panel 2×2 that de-confounds family from quality — driven by root `judge_bias.py` / `crossed_panel.py`), and `dimensionality.py` (construct validity / dimension redundancy: within-judge Cronbach α + PCA vs a cross-judge MTMM matrix that removes single-rater halo — driven by root `construct_probe.py`). `human_annotation_interface.py` is the Streamlit app.

### The six dimensions (defined once in `rubric.py`)

MD (misconception diagnosis), SS (scaffolding), ADR (answer-disclosure restraint), CMB (conceptual model building), TS (transfer success), PHA (pedagogical harm avoidance). All are 0–4 with justifications; composite is a weighted mean. Add/rename a dimension here and the judge prompt, scorecard, and validation pick it up automatically.

## Conventions and gotchas that span files

- **Run from the repo root.** The package is named `src` and everything imports as `from src.…`; there's no installed package alias.
- **Config is mostly not wired in.** The CLI reads only `config.student_simulator`. The `judge`, `conversation` (incl. `tutor_system_prompt`), `batch`, and `scoring_weights` sections of `config/default.yaml` are **not** consumed — those defaults live as Click option defaults and module constants (`DEFAULT_TUTOR_SYSTEM_PROMPT`, `rubric.DEFAULT_WEIGHTS`). Editing the YAML's weights or prompts will not change behavior until you thread them through. (`run-benchmark` now exposes `--student-model` and `--transfer-injection-offset` as CLI overrides.)
- **Disengagement detection is intent-not-substring.** `conversation_loop._check_disengagement` ends a dialogue on strong quit-phrases (word-boundary match) but treats filler words ("whatever", "never mind") as disengagement *only* when they dominate a short message — a bare substring match falsely killed ~⅔ of conversations with a fluent frontier student before the transfer turn.
- **Filenames are the contract between stages.** Scores are written as `{conversation_id}_score.json`; `score` treats every other `*.json` (except `run_metadata.json`/`run_summary.json`) as a conversation, and `load_scores` globs `*_score.json`. Don't rename these.
- **Scenario IDs encode topic.** `_make_scenario_id` produces `mech-/em-/thwv-/modq-` prefixes, and `Scorecard._infer_topic` parses that prefix for per-topic breakdowns. Keep the prefix scheme consistent.
- **Conversation loop keeps two role-swapped histories.** From the tutor's view student=`user`/tutor=`assistant`; from the student's view it's reversed. The tutor always gets the last word at `max_turns`.
- **Transfer is the outcome measure.** The student simulator is told to inject the `transfer_problem` `transfer_injection_offset` turns before the end (via an appended system-prompt instruction), and the student answers honestly from its *current* understanding — that exchange drives the TS score.
- **LLM JSON is regex-extracted.** Both the judge and the scenario generator pull the first `{…}` block out of the response, so prompt edits that change output shape can silently break parsing.
- **`loadenv.sh` bridges lowercase `.env` keys to SDK env vars.** The SDKs need uppercase `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`; `source loadenv.sh` maps the lowercase `.env` names onto them before a run.
- `results/` and `*.db` are git-ignored (runtime artifacts); `data/scenarios/` is generated and not committed.

## Published papers & docs (GitHub Pages)

`docs/` is served as a static site by **GitHub Pages — source: `main` branch, `/docs` path** → https://chaotic-curiosity-io.github.io/phys-tutor-bench/. The research write-ups live there as styled, self-contained HTML:
- `index.html` — v1 empirical report (local Ollama models)
- `frontier.html` — frontier-model dual-judge rerun (Opus 4.8 / Sonnet 4.6 / Haiku 4.5 / GPT-4o; judges Opus 4.8 + GPT-5.5); includes a worked example + a cost breakdown
- `methodology.html` — v2 PER-grounded methodology (§6 = the ten future research directions)
- `v3-plan.html` — v3 validity-bridge implementation plan
- `concept-paper.html` — fundable concept paper / prospectus for PER groups & funders
- `bridge-probe.html` — in-silico bridge probe: does tutoring process predict (simulated) transfer? cross-judge + range-restriction; built by `bridge_probe.py` from `docs/bridge_probe.json`
- `judge-bias.html` — judge self-preference: does an LLM judge favor its own model family? paired DiD on the dual-judge run; built by `judge_bias.py` from `docs/judge_bias.json`
- `construct-validity.html` — construct validity / redundancy: are the six dimensions distinct? within-judge α/PCA vs cross-judge MTMM; built by `construct_probe.py` from `docs/construct_probe.json`
- `transcripts/` — every saved conversation rendered to HTML (scenario context, dialogue, per-judge scores + justifications). **Generated** by `python build_transcripts.py` from `results/` + `data/scenarios/`. Because `results/` is gitignored, the rendered HTML under `docs/transcripts/` is the committed/published artifact — re-run the script and commit it whenever new runs are added.
- `research_corpus.json` — primary-source-verified evidence base behind the papers

Helper scripts (root): `build_transcripts.py` (render transcripts → `docs/transcripts/`), `docs_build_data.py` (rebuild `docs/` figures + `analysis_data.json`/`construct_validity.json` from scored results), `bridge_probe.py` (in-silico criterion-validity probe → `docs/bridge_probe.json` + `assets/bridge_scatter.png`), `judge_bias.py` (judge self-preference DiD → `docs/judge_bias.json` + `assets/judge_bias.png`), `construct_probe.py` (construct-validity / redundancy MTMM → `docs/construct_probe.json` + `assets/construct_validity.png`), `run_crossed.py` (**PAID** crossed-panel tutor run → `results/crossed/`; relabels the weak-prompt haiku so it doesn't collide with the strong one), `crossed_panel.py` (tier-controlled self-preference 2×2 → `docs/crossed_panel.json` + `assets/crossed_panel.png`), and `estimate_cost.py` (estimate a run's API cost from saved transcripts, no new calls; list prices, used for `frontier.html` §6).

**Publishing a new paper (standing rule):** add it to `docs/` as styled HTML (model new ones on `v3-plan.html` / `concept-paper.html` — shared CSS, nav backlinks top and bottom, `og:`/`twitter:` link-preview meta), cross-link it into the other papers' nav **and** the root `README.md`, then **commit to `main`** — Pages auto-builds; a feature branch would NOT deploy. Verify the live URL after the build settles.

**Keep READMEs in sync (standing rule):** whenever docs, commands, structure, or behavior change, update `README.md` (and any other READMEs) in the *same* change — never defer it to "later."

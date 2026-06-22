# PhysTutorBench

A benchmark system for evaluating how well generative AI models tutor introductory physics. Grounded in Physics Education Research (PER), it measures pedagogical quality across six validated dimensions — not whether an AI can solve physics, but whether it can *teach* it.

---

## 📚 Papers & Reports

All published via GitHub Pages at [`chaotic-curiosity-io.github.io/phys-tutor-bench`](https://chaotic-curiosity-io.github.io/phys-tutor-bench/).

| | Paper | What it is |
|:--:|-------|------------|
| **v1** | [**Local LLMs as physics tutors**](https://chaotic-curiosity-io.github.io/phys-tutor-bench/) | Empirical report — 4 local Ollama models × 12 PER scenarios scored on six pedagogical dimensions; judge κ = 0.73 |
| **★** | [**Frontier models as tutors (dual-judge rerun)**](https://chaotic-curiosity-io.github.io/phys-tutor-bench/frontier.html) | Opus 4.8 / Sonnet 4.6 / Haiku 4.5 / GPT-4o × the same 12 scenarios, scored by **two** frontier judges (Opus 4.8 + GPT-5.5). Every frontier tutor beats the best local model; cross-judge ρ = 0.72 but the same-family judge shows a ceiling effect |
| **📁** | [**All 118 transcripts**](https://chaotic-curiosity-io.github.io/phys-tutor-bench/transcripts/) | Every conversation from all three runs (v1 local, frontier, crossed panel), rendered in full: scenario context, turn-by-turn dialogue, and each judge's six scores with justifications |
| **🌉** | [**In-silico bridge probe**](https://chaotic-curiosity-io.github.io/phys-tutor-bench/bridge-probe.html) | Zero-cost dry run of the keystone question: do the five tutoring-*process* dimensions predict simulated **Transfer Success**? Circularity-broken (one judge's process vs the *other* judge's transfer) + range-restriction correction → process predicts transfer at **r ≈ 0.68–0.75** across 144 already-scored conversations. Motivates the v3 bridge study; does not replace it |
| **⚖️** | [**Judge self-preference**](https://chaotic-curiosity-io.github.io/phys-tutor-bench/judge-bias.html) | Does an LLM judge favor its own model family? On the dual-judge run the Opus judge appears to inflate Anthropic tutors by ~0.25 pts (p = 0.025) — **but a crossed-panel follow-up overturns it**: controlling for tutor quality the composite effect is null (tier-controlled **−0.04, p = 0.71**); the robust own-family bias is **localized to PHA** (+0.36), the most subjective dimension. A confounded finding corrected by a better design |
| **🔬** | [**Construct validity & redundancy**](https://chaotic-curiosity-io.github.io/phys-tutor-bench/construct-validity.html) | Are the six dimensions really six? Within a judge the rubric looks unidimensional (Cronbach α = **0.94**, PC1 = **79%**); a cross-judge multitrait-multimethod matrix removes halo → mean inter-dimension r drops **0.62 → 0.43**, dominant factor **79% → 54%**. Scaffolding/restraint/diagnosis are reliable & distinct (0.76–0.82); harm-avoidance fails to replicate across judges (**0.30**) — the same dimension that uniquely retains own-family judge bias after a crossed-panel correction |
| **🔀** | [**Crossed-panel correction**](https://chaotic-curiosity-io.github.io/phys-tutor-bench/crossed-panel.html) | The paid de-confounding follow-up to the self-preference study. With both families at both quality tiers (added gpt-5.5 + a weak-prompt haiku), the composite own-family effect **vanishes** (tier-controlled **−0.04, p = 0.71**) — it was a quality/ceiling confound. Only **PHA** retains it (+0.36). A worked case of a confounded finding corrected by a better design |
| **v2** | [**PER-grounded methodology**](https://chaotic-curiosity-io.github.io/phys-tutor-bench/methodology.html) | Validity critique of v1 + tiered methodology (process → measured learning gains) + annotated literature corpus + instantiable study kit |
| **v3** | [**Validity-bridge implementation plan**](https://chaotic-curiosity-io.github.io/phys-tutor-bench/v3-plan.html) | Preregisterable protocol — *does any automated tutoring score predict real learning?* Two bridge estimators, decision gates, phased roadmap |
| **📄** | [**Concept paper / prospectus**](https://chaotic-curiosity-io.github.io/phys-tutor-bench/concept-paper.html) | Fundable, PER-publication-grade prospectus for the validity-bridge study; honest novelty vs. recent AI-tutor RCTs; phased funding case |
| **🗂** | [**Research corpus**](https://chaotic-curiosity-io.github.io/phys-tutor-bench/research_corpus.json) `JSON` | Primary-source-verified evidence base (PER measurement, tutoring science, LLM-eval, funding) behind the papers |

---

## 📄 Live report — Local open-weight LLMs as physics tutors

### ▶ **[Read the report](https://chaotic-curiosity-io.github.io/phys-tutor-bench/)** · `chaotic-curiosity-io.github.io/phys-tutor-bench`

A study using this benchmark to evaluate four local [Ollama](https://ollama.com) models as introductory-physics tutors — judged on *teaching* quality, not problem-solving. 12 PER-grounded scenarios × 4 models = 48 conversations, each scored on all six dimensions with justifications.

| Rank | Model | Composite&nbsp;/4 |
|:----:|-------|:----------------:|
| 🥇 | **qwen3:8b** | **2.38** |
| 🥈 | llama3.1:8b | 1.82 |
| 🥉 | qwen2.5:7b | 1.48 |
| 4 | llama3.2:3b | 1.01 |

**Headlines:** answer-disclosure restraint is the *universal* weak spot — every model tends to lecture the answer rather than guide. `llama3.2:3b` frequently reinforces the student's misconception or states incorrect physics (harm-avoidance 0.08/4). Judge test–retest reliability is substantial (quadratic-weighted κ = 0.73; 92% of scores within one point). Full methods, figures, per-topic breakdowns, construct/judge validity, and threats-to-validity are in the report.

Reproduce: `run_local_eval.py` → judge (`judge_rubric.md`) → `finalize_scores.py` → `docs_build_data.py`; reliability via `compute_agreement.py` (see report §8).

### 🚀 Companion: **[Frontier models as physics tutors — a dual-judge rerun](https://chaotic-curiosity-io.github.io/phys-tutor-bench/frontier.html)**

The same protocol with frontier models in every role: **Claude Opus 4.8, Sonnet 4.6, Haiku 4.5, and GPT-4o** as tutors, a fixed **Claude Sonnet 4.6** student, and **two independent judges** (Claude Opus 4.8 + GPT-5.5) scoring all 48 conversations.

| Tutor | Opus 4.8 judge | GPT-5.5 judge |
|-------|:--:|:--:|
| 🥇 claude-opus-4-8 | **4.00** | **3.70** |
| 🥈 claude-sonnet-4-6 | 3.92 | 3.68 |
| 🥉 claude-haiku-4-5 | 3.84 | 3.45 |
| 4 · gpt-4o | 2.96 | 2.90 |

**Headlines:** every frontier tutor beats v1's best local model (qwen3:8b, 2.38); **answer-disclosure restraint stays the universal weak spot** (GPT-4o worst, 1.75 / 1.25); GPT-4o's gap is *scaffolding*, not diagnosis. The two judges agree on the **ranking** (composite Spearman ρ = 0.72) but the same-family Opus judge rates Claude tutors at the **ceiling** (a perfect 4.00 for the Opus tutor) while cross-family GPT-5.5 is stricter — a self-preference signal the dual-judge design exists to catch. Full tables, figures, a worked example, inter-judge agreement, and threats-to-validity in the report.

All 96 conversations (this run's 48 + v1's 48) are **published as full transcripts** → [browse them](https://chaotic-curiosity-io.github.io/phys-tutor-bench/transcripts/). The frontier run cost **~$18** in API usage (estimated, list prices; v1 was ~$0 — local models + a Claude Code subscription).

### 🌉 Companion: **[In-silico bridge probe — does tutoring process predict transfer?](https://chaotic-curiosity-io.github.io/phys-tutor-bench/bridge-probe.html)**

The benchmark's funding case rests on an unproven claim — that an automated *process* score predicts real *learning*. The definitive test needs human pre/post gains (the [v3 bridge study](https://chaotic-curiosity-io.github.io/phys-tutor-bench/v3-plan.html)). But the benchmark already scores a *simulated* outcome — **Transfer Success** — so the bridge's logic can be run in-silico for **$0** on the 144 conversations already scored. The naive process↔transfer correlation is circular (one judge scores both from one transcript); we break it by regressing **one judge's process composite against the *other* judge's transfer score**. Circularity-broken and range-restricted it is **r = 0.37** (ρ = 0.52); corrected for range restriction **≈ 0.68**; an independent wide-range estimate **0.75** — two methods with opposite biases converging on **r ≈ 0.7**. Process quality here is not merely stylistic; it tracks the (simulated) outcome across independent raters. Honest caveat: simulated transfer ≠ measured human learning, so this *motivates* the bridge study, it doesn't substitute for it. Reproduce: `python bridge_probe.py` (no new API calls).

### ⚖️ Companion: **[Judge self-preference — does an LLM judge favor its own family?](https://chaotic-curiosity-io.github.io/phys-tutor-bench/judge-bias.html)**

The dual-judge design exists to catch a judge over-rewarding its own kind. It does. A paired **difference-in-differences** on the within-conversation score gap (Opus judge − GPT-5.5 judge) cancels conversation difficulty and overall leniency; what remains is family-aligned disagreement. Result: relative to the cross-family GPT judge, the Opus judge inflates **Anthropic** tutors by **+0.31** composite points but the lone OpenAI tutor by only **+0.06** — a DiD of **+0.25** (permutation p = 0.025, 95% CI [+0.05, +0.46]). The mechanism is a ceiling: the Opus judge rates **93.5%** of its own family's dimensions at the maximum 4 and hands the Opus *tutor* a literal 4.000. The cleanest evidence it's genuine favoritism and not generic generosity: of the six dimensions, five judge the *tutor* (DiD +0.17 to +0.64) and one — Transfer Success — judges the *fixed student model* and shows **−0.08, none**. Honest limit: family is confounded with quality in this panel (all Anthropic tutors strong, the one OpenAI tutor weak), so the magnitude is an upper bound and the effect is one-sided; a crossed panel is the fix. Reproduce: `python judge_bias.py` (no new API calls).

**Update — the crossed panel (paid run; full [standalone report](https://chaotic-curiosity-io.github.io/phys-tutor-bench/crossed-panel.html), also [§9 of the page](https://chaotic-curiosity-io.github.io/phys-tutor-bench/judge-bias.html#crossed)).** We ran the fix: added a strong OpenAI tutor (gpt-5.5) and a weak Anthropic one (haiku-4-5 + an answer-dumper prompt) so family is no longer confounded with quality, giving a family × tier 2×2 over 70 conversations. **The composite self-preference effect does not survive** — tier-controlled **−0.04 (p = 0.71)**; the original +0.25 was largely a ceiling/quality interaction (the Opus judge is even *harsher* than GPT on the weak Anthropic tutor). What survives is **PHA-specific (+0.36)**: of the six dimensions, only pedagogical-harm-avoidance retains own-family bias after tier control — the same dimension the construct-validity audit found least reliable (0.30). A worked case of a plausible finding corrected by a stronger design. Reproduce: `python run_crossed.py` (paid) → `python crossed_panel.py`.

### 🔬 Companion: **[Construct validity — are the six dimensions really six?](https://chaotic-curiosity-io.github.io/phys-tutor-bench/construct-validity.html)**

A redundancy audit of the rubric. Within a single judge the six dimensions look like one factor: Cronbach **α = 0.94**, first principal component **79%** of variance, six dimension-pairs above r = 0.8 (MD–CMB = 0.93). But one judge rating all six from one transcript manufactures **halo**. A cross-judge **multitrait-multimethod** matrix (one judge's dimension vs the *other* judge's) removes it: on the same 48 conversations, mean inter-dimension correlation drops **0.62 → 0.43** and the dominant factor **79% → 54%**, while same-dimension cross-judge agreement beats cross-dimension on **52/60** Campbell–Fiske comparisons. The rubric carries real multidimensional signal — unevenly: SS/ADR/MD are reliable & distinct (cross-judge **0.76–0.82**), CMB is weak (0.49), and **PHA fails to replicate (0.30)** — the very dimension with the largest self-preference inflation in the [judge-bias study](https://chaotic-curiosity-io.github.io/phys-tutor-bench/judge-bias.html). Reproduce: `python construct_probe.py` (no new API calls).

### 🧪 Companion: **[PhysTutorBench v2 — a PER-grounded methodology](https://chaotic-curiosity-io.github.io/phys-tutor-bench/methodology.html)**

A critique of this v1 study against educational-measurement standards, an annotated PER + LLM-eval literature corpus, and a superseding **tiered methodology** — moving from LLM-judged *process quality* to **measured student learning gains** as the criterion, with a practical LLM tier whose validity is *bridged* to the gold standard. Includes an instantiable item/rubric schema and a preregistration checklist for PER groups.

### 🧭 Next: **[v3 implementation plan — the validity-bridge study](https://chaotic-curiosity-io.github.io/phys-tutor-bench/v3-plan.html)**

A preregisterable study protocol testing the keystone question — *do automated tutoring scores predict real student learning gains?* — with a worked mechanics/FCI instantiation, two bridge estimators, decision gates, and a phased roadmap sequencing all ten research directions.

### 📄 For PER groups & funders: **[Concept paper — Do automated tutoring-quality metrics predict real physics learning?](https://chaotic-curiosity-io.github.io/phys-tutor-bench/concept-paper.html)**

A PER-publication-grade research prospectus built to survive a hostile measurement, statistics, ethics, and funding review. Hardens the v3 plan with an honest novelty position against the recent AI-tutor learning-gain RCTs (Kestin 2025; Tutor CoPilot 2024), an IRT latent-change criterion, a disattenuated cross-validated decision gate, first-class equity, and a phased, primary-source-verified funding case (NSF RITEL/IUSE, Cottrell seed; Gates/Digital Promise external validation). Evidence base assembled by a multi-agent research → adversarial-critique → verification harness.

---

## What It Measures

PhysTutorBench runs any LLM through simulated tutoring conversations where a student exhibits specific misconceptions catalogued by validated PER instruments (FCI, CSEM, BEMA, etc.), then scores the tutor on:

| Dimension | Abbr | What It Measures |
|---|---|---|
| Misconception Diagnosis | MD | Does the tutor identify the *specific* misconception? |
| Scaffolding Strategy | SS | Does the tutor scaffold interactively, matched to the student's ZPD? |
| Answer Disclosure Restraint | ADR | Does the tutor avoid giving away the answer? |
| Conceptual Model Building | CMB | Does the tutor help build a *transferable* mental model? |
| Transfer Success | TS | Can the student apply the concept to a novel problem? |
| Pedagogical Harm Avoidance | PHA | Does the tutor avoid reinforcing misconceptions, shaming, or incorrect physics? |

Each dimension is scored 0-4 with natural-language justifications referencing specific conversation moments.

## Architecture

```
Scenario Bank          Conversation Engine         Scoring           Validation
(misconception         (student simulator +        (LLM judge +      (human annotation +
 taxonomy +             tutor under test +          rubric +           agreement analysis +
 generated              batch runner)               scorecard)         construct/discriminant
 scenarios)                                                            validity)
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Generate scenarios from the misconception taxonomy
phystutor generate-scenarios --topic mechanics --count 3

# Run a single scenario against a model
phystutor run-single --scenario data/scenarios/mechanics/some-scenario.json --model claude-opus-4-8

# Run the full benchmark. Frontier models (Opus 4.8/4.7, Fable 5, OpenAI o-series/GPT-5.x)
# that reject `temperature` are handled automatically by src/engine/model_compat.py.
# --student-model overrides the fixed student; --transfer-injection-offset matches a prior run.
phystutor run-benchmark --model claude-opus-4-8 --student-model claude-sonnet-4-6 \
  --max-turns 10 --transfer-injection-offset 3 --concurrency 6 --output results/frontier

# Score the conversations. --judge-model dispatches by prefix to the Anthropic LLMJudge
# or the OpenAIJudge; --output keeps each judge's scores in its own directory.
phystutor score --results-dir results/frontier --judge-model claude-opus-4-8 --output results/frontier/_scores/opus-4-8
phystutor score --results-dir results/frontier --judge-model gpt-5.5         --output results/frontier/_scores/gpt-5.5

# Generate comparison scorecard (point at one judge's score dir)
phystutor scorecard --results-dir results/frontier/_scores/opus-4-8 --compare claude-opus-4-8,claude-sonnet-4-6,claude-haiku-4-5,gpt-4o

# Inter-judge agreement when two judges scored the same conversations
python compare_judges.py --a results/frontier/_scores/opus-4-8 --b results/frontier/_scores/gpt-5.5

# Publish: render every saved conversation to browsable HTML (docs/transcripts/)
python build_transcripts.py

# Estimate API cost of a run from saved transcripts (no new calls)
python estimate_cost.py

# In-silico validity-bridge probe: does process predict (simulated) transfer? (no new calls)
python bridge_probe.py

# Judge self-preference: does each judge over-reward its own model family? (no new calls)
python judge_bias.py

# Construct validity / redundancy: are the six dimensions distinct, or one factor? (no new calls)
python construct_probe.py

# Crossed-panel run to de-confound judge family from tutor quality (PAID: real API calls)
python run_crossed.py        # adds gpt-5.5 + a weak-prompt haiku tutor, then score with both judges
python crossed_panel.py      # tier-controlled 2x2 self-preference analysis (no new calls)

# Launch human annotation interface
phystutor annotate --conversations results/sample/

# Run validation analyses
phystutor validate --annotations data/validation/human_annotations.db --results-dir results/
```

## Project Structure

```
phystutor-bench/
├── config/default.yaml              # Default configuration
├── data/
│   ├── taxonomies/
│   │   └── misconception_taxonomy.json   # 65+ misconceptions from PER instruments
│   ├── scenarios/                    # Generated tutoring scenarios
│   └── validation/                   # Gold/foil conversations, annotations
├── src/
│   ├── scenarios/                    # Schema, taxonomy, generation
│   ├── engine/                       # Student simulator, tutor runner, conversation loop
│   ├── scoring/                      # Rubric, LLM judge, scorecard
│   ├── validation/                   # Annotation interface, agreement, validity
│   └── cli.py                        # CLI entry point
├── tests/                            # 127 tests across all subsystems
└── results/                          # Generated at runtime
```

## Misconception Taxonomy

The taxonomy includes 65+ misconceptions from validated PER instruments:

- **Mechanics** (~30): From FCI, FMCE, MBT — kinematics, Newton's laws, forces, energy, momentum, gravity, rotation
- **Electricity & Magnetism** (~20): From CSEM, BEMA, DIRECT — electrostatics, fields, circuits, magnetism, induction
- **Thermal/Waves** (~15): From TCS, MWCS, FTGOT — heat/temperature, wave properties, optics
- **Modern/Quantum** (~13): From QMCS, QMVI — wave-particle duality, uncertainty, quantum states, relativity

## Design Principles

- **Student simulator is a controlled variable** — fixed model + prompt across all evaluations
- **Misconception taxonomy is the backbone** — every scenario traces to a validated PER instrument
- **Transfer questions are the outcome measure** — not whether the student parrots the answer, but whether they can apply it
- **Scores need justifications** — the judge explains its reasoning, referencing specific conversation moments
- **Reproducibility** — all model versions, temperatures, and system prompts are logged

## Requirements

- Python 3.11+
- Anthropic API key (for scenario generation, student simulation, and LLM judge)
- Optional: OpenAI API key (for evaluating OpenAI models as tutors)

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT

# PhysTutorBench

A benchmark system for evaluating how well generative AI models tutor introductory physics. Grounded in Physics Education Research (PER), it measures pedagogical quality across six validated dimensions — not whether an AI can solve physics, but whether it can *teach* it.

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

### 🧪 Companion: **[PhysTutorBench v2 — a PER-grounded methodology](https://chaotic-curiosity-io.github.io/phys-tutor-bench/methodology.html)**

A critique of this v1 study against educational-measurement standards, an annotated PER + LLM-eval literature corpus, and a superseding **tiered methodology** — moving from LLM-judged *process quality* to **measured student learning gains** as the criterion, with a practical LLM tier whose validity is *bridged* to the gold standard. Includes an instantiable item/rubric schema and a preregistration checklist for PER groups.

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
phystutor run-single --scenario data/scenarios/mechanics/some-scenario.json --model claude-sonnet-4-20250514

# Run the full benchmark
phystutor run-benchmark --model claude-sonnet-4-20250514 --concurrency 5

# Score the conversations
phystutor score --results-dir results/claude-sonnet-4-20250514/2025-01-15_120000/

# Generate comparison scorecard
phystutor scorecard --results-dir results/ --compare claude-sonnet-4-20250514,gpt-4o

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
├── tests/                            # 71 tests across all subsystems
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

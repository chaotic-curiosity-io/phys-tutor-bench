# Claude Code Starting Prompt: Physics Tutoring Benchmark (PhysTutorBench)

## What You're Building

Build **PhysTutorBench**, a benchmark system that evaluates how well generative AI models tutor introductory physics. It takes any LLM, runs it through a battery of simulated physics tutoring conversations, and produces a validated scorecard of its pedagogical quality across multiple dimensions.

This is NOT a benchmark for whether an AI can solve physics problems. It is a benchmark for whether an AI can TEACH physics — specifically, whether it can diagnose student misconceptions, scaffold understanding without giving away answers, help students build transferable mental models, and avoid pedagogical harm.

The benchmark is grounded in Physics Education Research (PER) — a mature field with 30+ years of validated instruments (concept inventories), misconception taxonomies, and pedagogical frameworks. This grounding is what differentiates PhysTutorBench from existing AI tutoring benchmarks (TutorBench, MathTutorBench, SafeTutors, GuideEval), which are built by ML researchers and treat physics as interchangeable with math word problems.

---

## Research Foundation (What You Need to Know)

### Existing Benchmarks and Their Gaps

The following benchmarks exist and partially overlap with this project. PhysTutorBench fills gaps they leave open:

1. **TutorBench** (Scale AI, Oct 2025): 1,490 conversations across 6 STEM subjects (including physics). High school/AP level. Three use cases: adaptive explanation, feedback/assessment, active learning support. Uses sample-specific rubrics (15,220 total) graded by an LLM judge. Finding: no frontier model exceeds 56% pass rate. GAP: Physics is diluted among 6 subjects; rubrics are not grounded in PER misconception taxonomies; no transfer assessment.

2. **MathTutorBench** (Macina et al., Feb 2025): Math-only. Trains a reward model (Qwen2.5-1.5B) to score scaffolding quality. Finds that domain expertise ≠ teaching ability; pedagogy and subject mastery form a trade-off. GAP: Math-only; no physics misconception structure.

3. **SafeTutors** (March 2025): Jointly evaluates safety and pedagogy across math/physics/chemistry. Defines 11 harm dimensions and 48 sub-risks. Key insight: tutoring safety ≠ conventional LLM safety. The risk is "quiet erosion of learning through answer over-disclosure, misconception reinforcement, and abdication of scaffolding." GAP: Multi-turn but doesn't measure conceptual change; physics coverage is thin.

4. **GuideEval** (Aug 2025): Evaluates Socratic LLMs on perception/orchestration/elicitation. Finds models fail to adapt scaffolding when learners show confusion. GAP: Not physics-specific; doesn't use validated PER instruments.

5. **SocraticLM** (NeurIPS 2024): Uses Dean-Teacher-Student pipeline with 6 simulated student cognitive states. Math-focused training data. GAP: No physics grounding.

6. **MRBench** (Dec 2024): 192 conversations, 1,596 responses, 8 pedagogical dimensions. Math-only. Uses Prometheus2 as LLM judge (with poor results). GAP: Small scale, math-only.

### Physics Education Research Instruments

These are the validated assessment tools from PER that provide the misconception taxonomy backbone:

**Mechanics:**
- Force Concept Inventory (FCI) — Hestenes, Wells, Swackhamer 1992. 30 MC questions. Probes ~30 named misconceptions across: Kinematics (position-velocity conflation, nonvectorial velocity composition), Impetus (hit-supplied impetus, impetus dissipation, circular impetus), Active Force (only active agents exert forces, motion implies active force, no motion implies no force, velocity proportional to applied force, acceleration implies increasing force), Action/Reaction (greater mass = greater force, most active agent produces greatest force), Concatenation of Influences (last force determines motion, force compromise determines motion). The FCI misconception taxonomy (Table II of the original paper) is the single most validated misconception structure in physics education.
- Force and Motion Conceptual Evaluation (FMCE) — Thornton & Sokoloff 1998. Complements FCI with emphasis on graphical interpretation of kinematics.
- Mechanics Baseline Test (MBT) — Hestenes & Wells 1992. Tests quantitative problem-solving in mechanics.

**Electricity & Magnetism:**
- Conceptual Survey of Electricity and Magnetism (CSEM) — Maloney et al. 2001. 32 items covering electrostatics, circuits, magnetism.
- Brief Electricity and Magnetism Assessment (BEMA) — Ding et al. 2006. 30 items, more emphasis on Gauss's law, Faraday's law.
- Determining and Interpreting Resistive Electric Circuit Concepts Test (DIRECT) — Engelhardt & Beichner 2004. Circuit-specific misconceptions.

**Thermal/Waves/Optics:**
- Thermal Concepts Survey (TCS) — Wattanakasiwich et al. 2013.
- Mechanical Waves Conceptual Survey (MWCS) — Tongchai et al. 2009.
- Four-Tier Geometrical Optics Test (FTGOT) — Kaltakci-Gurel et al. 2017.

**Quantum:**
- Quantum Mechanics Conceptual Survey (QMCS) — McKagan et al. 2010.
- Quantum Mechanics Visualization Instrument (QMVI) — Cataloglu & Robinett 2002.

**Key insight for implementation:** You do NOT need the actual test items (many are restricted/copyrighted). You need the MISCONCEPTION TAXONOMIES — the categorized lists of what students get wrong and why. These are published in the associated papers and are the structural backbone of your scenario bank.

### Pedagogical Frameworks for Scoring

These frameworks define what "good tutoring" means and should ground your rubric dimensions:

- **Bloom's 2-Sigma Problem** (Bloom, 1984): One-on-one tutoring produces 2 standard deviations of improvement over lecture. The benchmark should measure whether AI approaches this.
- **Clement's Bridging Analogies** (Clement, 1993): Expert physics tutors use "anchoring conceptions" (intuitions the student already has that are correct) and build conceptual bridges to the target concept. A good tutor doesn't start from scratch — they find what the student already understands correctly.
- **Chi's ICAP Framework** (Chi & Wylie, 2014): Learning activities ranked: Interactive > Constructive > Active > Passive. Good tutoring should push students toward interactive/constructive engagement, not passive reception.
- **Minstrell's Facets of Understanding** (Minstrell, 1992): Student understanding exists as "facets" — partial understandings that mix correct and incorrect elements. Good tutoring works with facets, not against them.
- **Hake's Normalized Gain** (Hake, 1998): <g> = (post - pre) / (100 - pre). Standard measure of instructional effectiveness in PER. Your transfer question is the analog of post-test.
- **Vygotsky's Zone of Proximal Development**: Scaffolding should target what the student can do with help but not alone. Over-scaffolding (giving too much) and under-scaffolding (asking questions they can't possibly answer) are both failures.

---

## System Architecture

PhysTutorBench has four subsystems. Build them in this order:

### Subsystem 1: Scenario Bank (`/scenarios`)

Each scenario is a JSON object:

```jsonc
{
  "id": "mech-n3l-ar-001",
  "topic_area": "mechanics",
  "subtopic": "newton_third_law",
  "level": "intro_algebra",  // intro_algebra | intro_calculus | upper_division
  
  // The tutoring situation
  "problem_context": "A 1000 kg car collides with a 100 kg motorcycle. The student is asked to compare the forces each vehicle exerts on the other during the collision.",
  "student_initial_response": "The car exerts way more force on the motorcycle because it's bigger and heavier. That's why the motorcycle gets crushed.",
  
  // Misconception grounding
  "misconception_tags": ["AR1_greater_mass_greater_force"],
  "misconception_source": "FCI",
  "misconception_description": "Student believes that larger/more massive objects exert greater forces in interactions, violating Newton's Third Law.",
  "common_prevalence": "high",  // from PER literature
  
  // Student profile
  "student_profile": {
    "knowledge_level": "intro_algebra",
    "knows": ["free_body_diagrams_basic", "newton_second_law_qualitative"],
    "struggles_with": ["distinguishing_force_from_effect", "vector_nature_of_force"],
    "affect": "confident_but_wrong",
    "response_style": "verbose_reasoner"
  },
  
  // Transfer assessment
  "transfer_problem": "Now consider an astronaut on a spacewalk who pushes against the International Space Station. Compare the forces. What happens to each?",
  "transfer_success_criteria": "Student states forces are equal and opposite, and correctly explains the different accelerations are due to mass differences (a = F/m), applying this to the new context without prompting.",
  
  // Expert reference (for validation, not shown to models under test)
  "expert_strategy": {
    "diagnosis": "AR1 — student conflates force magnitude with effect/damage",
    "recommended_approach": "bridging_analogy",
    "scaffolding_sequence": [
      "Acknowledge the intuition that the motorcycle gets more damaged (this is correct and is an anchoring conception)",
      "Distinguish between force and effect/acceleration — same force, different masses, different accelerations",
      "Use Newton's 2nd law to show why equal forces produce unequal effects",
      "Bridge to the general principle: interaction forces are always equal and opposite",
      "Test with transfer problem"
    ],
    "common_pitfalls": [
      "Telling the student 'Newton's Third Law says forces are equal' without addressing WHY their intuition about damage is actually partly correct",
      "Skipping the force vs. effect distinction, which is the actual conceptual gap",
      "Using only mathematical derivation without physical reasoning"
    ]
  }
}
```

**Build 200-300 scenarios** organized as:
- Mechanics: ~100 (FCI/FMCE misconception clusters)
- Electricity & Magnetism: ~60 (CSEM/BEMA clusters)  
- Thermal/Waves: ~40 (TCS/MWCS clusters)
- Modern/Quantum: ~30 (QMCS clusters)

For the initial implementation, **generate these using a frontier LLM** (Claude Opus or Sonnet) prompted with the misconception taxonomies from each instrument. Each generated scenario should cite which specific misconception tag it targets. These will later be validated by human PER experts, but the LLM-generated versions are the starting corpus.

Create a Python script `generate_scenarios.py` that:
1. Takes a misconception taxonomy (structured list of misconception IDs, descriptions, and associated concepts) as input
2. Generates scenario JSON objects using the Anthropic API
3. Validates schema compliance
4. Outputs to `/scenarios/{topic_area}/`

Also create `misconception_taxonomy.json` — the master taxonomy file that compiles misconception IDs, descriptions, source instruments, topic mappings, and known prevalence data from ALL the instruments listed above. This is the knowledge backbone of the entire system.

### Subsystem 2: Conversation Engine (`/engine`)

This orchestrates the simulated tutoring interaction.

**Components:**

`student_simulator.py` — An LLM-based simulated student. Takes a scenario's student profile and misconception as input. System prompt instructs it to:
- Hold the specified misconception and reason from it consistently
- Respond at the specified knowledge level and affect
- NOT be trivially convinced — require actual scaffolding to shift understanding
- Show realistic intermediate states (partial understanding, new questions, occasional backsliding)
- When presented with the transfer problem: attempt it genuinely based on current understanding (not just agree with the tutor)

The student simulator should use a FIXED model (e.g., Claude Sonnet with a locked system prompt and temperature) across all evaluations. It is a controlled variable.

`tutor_runner.py` — Manages the model under test. Takes:
- A system prompt that establishes the tutoring context (but does NOT include the scenario's expert strategy — the model must figure out its own approach)
- The student's initial response from the scenario
- Configuration for the model under test (API endpoint, model name, parameters)

`conversation_loop.py` — Orchestrates the multi-turn interaction:
1. Presents the problem context and student's initial response to the tutor
2. Tutor responds
3. Student simulator responds (conditioned on full conversation history + its misconception profile)
4. Repeat until termination condition:
   - Max turns reached (configurable, default 15)
   - Student demonstrates conceptual change on the original problem AND the transfer problem is injected
   - Student explicitly disengages
5. At turn N-2 (2 turns before max), inject the transfer problem via the student simulator: "Okay, I think I get it. But what about [transfer problem]?"
6. Record full conversation transcript with metadata

Output: `conversation_record.json` containing full transcript, timing, token counts, model metadata, scenario reference, and termination reason.

`batch_runner.py` — Runs the full scenario bank against one or more models under test. Handles:
- Parallel execution (configurable concurrency)
- Progress tracking and resumption
- Cost estimation before run
- Output organization: `/results/{model_name}/{run_timestamp}/`

### Subsystem 3: Scoring Rubric (`/scoring`)

The judge evaluates each conversation on 6 dimensions. Each dimension produces a score from 0-4.

**Dimension 1: Misconception Diagnosis (MD)**
- 0: Tutor does not identify any misconception; treats the student as simply "wrong"
- 1: Tutor identifies the student is wrong but misdiagnoses the specific misconception
- 2: Tutor implicitly addresses the correct misconception but doesn't explicitly surface it
- 3: Tutor correctly identifies the misconception category and addresses it directly
- 4: Tutor identifies the specific misconception, names it or describes it precisely, and explains why it's a common/natural way to think (validating the student's reasoning process while correcting the conclusion)

**Dimension 2: Scaffolding Strategy (SS)**
- 0: Tutor lectures / dumps information with no interactive structure
- 1: Tutor asks questions but they're leading/rhetorical (pseudo-Socratic)
- 2: Tutor uses genuine scaffolding but mismatched to the student's ZPD (too advanced or too basic)
- 3: Tutor uses appropriate scaffolding strategy well-matched to the student's current understanding
- 4: Tutor uses bridging from anchoring conceptions, calibrates to the student's ZPD, and adjusts strategy based on student responses (adaptive scaffolding)

**Dimension 3: Answer Disclosure Restraint (ADR)**
- 0: Tutor immediately gives the correct answer and explanation
- 1: Tutor gives hints that are effectively the answer with minimal disguise
- 2: Tutor mostly guides but slips into telling at key moments
- 3: Tutor consistently guides without disclosing, student constructs understanding
- 4: Tutor maintains productive struggle — student arrives at the answer through their own reasoning with minimal nudging

**Dimension 4: Conceptual Model Building (CMB)**
- 0: Tutor focuses only on getting the right numerical/factual answer
- 1: Tutor explains the concept but in isolated/memorizable form ("Newton's Third Law says...")
- 2: Tutor connects the concept to the specific problem but doesn't generalize
- 3: Tutor helps student build a generalizable mental model that extends beyond the specific problem
- 4: Tutor explicitly connects the concept to the student's existing correct knowledge, builds a transferable model, and helps the student see when/where it applies and when it doesn't (boundary conditions)

**Dimension 5: Transfer Success (TS)**
- 0: Transfer problem not reached (conversation hit max turns without progress) OR student fails transfer completely
- 1: Student attempts transfer but applies the misconception again
- 2: Student partially succeeds on transfer — gets the principle but can't fully apply it
- 3: Student succeeds on transfer with minor prompting from the tutor
- 4: Student independently and correctly applies the concept to the novel transfer context

**Dimension 6: Pedagogical Harm Avoidance (PHA)**
- 0: Tutor actively reinforces the misconception or introduces new ones or uses incorrect physics
- 1: Tutor uses correct physics but makes pedagogically harmful moves (shaming, overwhelming, creating learned helplessness)
- 2: Tutor is neutral — correct but not actively harmful or helpful
- 3: Tutor avoids harm and maintains student engagement/motivation
- 4: Tutor actively builds student confidence, validates productive aspects of their reasoning, and maintains a safe learning environment throughout

**Implementation:**

`judge.py` — Takes a conversation record and scenario, produces scores on all 6 dimensions with justifications.

Two judge backends:
1. `llm_judge.py` — Uses a frontier LLM (Claude Opus or GPT-4o) prompted with the rubric definitions, scenario context (including misconception tags and expert strategy for reference), and full conversation transcript. Outputs structured JSON with scores and natural-language justifications for each dimension.
2. `reward_model_judge.py` — (Future: once human annotations are available) A fine-tuned smaller model that produces scores directly. Stub this out with the interface but don't implement training yet.

`scorecard.py` — Aggregates scores across all scenarios for a model run. Produces:
- Overall composite score (weighted average across dimensions)
- Per-dimension averages
- Per-topic-area breakdowns (mechanics, E&M, thermal, modern)
- Per-misconception-cluster performance
- Comparison tables across models
- Distribution plots for each dimension

### Subsystem 4: Validation Framework (`/validation`)

This establishes that the benchmark actually measures what it claims.

`human_annotation_interface.py` — A simple web interface (Streamlit or similar) that presents conversation transcripts to human raters and collects scores on all 6 dimensions. Features:
- Displays the scenario context, misconception tags, and expert strategy as reference
- Shows the full conversation transcript
- Provides the rubric definitions inline
- Collects 0-4 scores per dimension with optional free-text justification
- Tracks annotator ID for inter-rater reliability analysis

`agreement_analysis.py` — Computes inter-rater reliability:
- Cohen's kappa (pairwise between annotators)
- Krippendorff's alpha (multi-annotator)
- Spearman's rho per dimension (ordinal correlation)
- Confusion matrices per dimension

`construct_validity.py` — Runs analyses to establish construct validity:
- Correlation between MD score and SS appropriateness (if you diagnose correctly, do you scaffold better?)
- Correlation between ADR score and TS score (does restraint predict transfer success?)
- Correlation between CMB and TS (does model-building predict transfer?)
- Factor analysis across the 6 dimensions to check they measure distinct constructs

`discriminant_validity.py` — Tests with known-quality conversations:
- Generate "gold standard" conversations (expert tutoring) — should score 3-4 across all dimensions
- Generate "answer dumper" conversations (tutor just explains the answer) — should score high on MD/PHA but low on ADR/SS/CMB
- Generate "wrong physics" conversations (confident but incorrect tutor) — should score low on PHA
- Generate "pseudo-Socratic" conversations (asks questions but they're rhetorical leading questions) — should score low on SS/ADR
- Verify the benchmark ranks these in the expected order

---

## Tech Stack

- **Language:** Python 3.11+
- **LLM Integration:** Anthropic SDK (`anthropic` package) as primary; OpenAI SDK and generic HTTP for other models
- **Data:** JSON for scenarios and conversation records; SQLite for annotation data and results aggregation
- **Visualization:** Matplotlib/Seaborn for scorecards and analysis plots; optionally Plotly for interactive dashboards
- **Annotation Interface:** Streamlit
- **CLI:** Click or Typer for the command-line interface
- **Testing:** pytest with fixtures for each subsystem

---

## Project Structure

```
phystutor-bench/
├── README.md
├── pyproject.toml
├── config/
│   ├── default.yaml          # Default configuration (models, parameters, paths)
│   └── models/               # Per-model configuration files
├── data/
│   ├── taxonomies/
│   │   └── misconception_taxonomy.json
│   ├── scenarios/
│   │   ├── mechanics/
│   │   ├── em/
│   │   ├── thermal_waves/
│   │   └── modern_quantum/
│   └── validation/
│       ├── gold_conversations/
│       └── foil_conversations/
├── src/
│   ├── __init__.py
│   ├── scenarios/
│   │   ├── generate_scenarios.py
│   │   ├── schema.py          # Pydantic models for scenario validation
│   │   └── taxonomy.py        # Taxonomy loading and querying
│   ├── engine/
│   │   ├── student_simulator.py
│   │   ├── tutor_runner.py
│   │   ├── conversation_loop.py
│   │   └── batch_runner.py
│   ├── scoring/
│   │   ├── judge.py
│   │   ├── llm_judge.py
│   │   ├── reward_model_judge.py  # Stub
│   │   ├── rubric.py          # Rubric definitions as structured data
│   │   └── scorecard.py
│   ├── validation/
│   │   ├── human_annotation_interface.py
│   │   ├── agreement_analysis.py
│   │   ├── construct_validity.py
│   │   └── discriminant_validity.py
│   └── cli.py                 # Main CLI entry point
├── tests/
│   ├── test_scenarios.py
│   ├── test_engine.py
│   ├── test_scoring.py
│   └── test_validation.py
└── results/                   # Generated at runtime
```

---

## CLI Interface

The CLI should support these commands:

```bash
# Generate scenarios from taxonomy
phystutor generate-scenarios --topic mechanics --count 50

# Run a single scenario against a model (useful for development/debugging)
phystutor run-single --scenario mech-n3l-ar-001 --model claude-sonnet-4-20250514

# Run full benchmark suite against a model
phystutor run-benchmark --model claude-sonnet-4-20250514 --concurrency 5

# Score a set of conversation records
phystutor score --results-dir results/claude-sonnet/2025-01-15/

# Generate scorecard report
phystutor scorecard --results-dir results/ --compare claude-sonnet,gpt-4o,gemini-pro

# Launch annotation interface for human validation
phystutor annotate --conversations results/sample_for_annotation/

# Run validation analyses
phystutor validate --annotations data/validation/human_annotations.db
```

---

## Implementation Order

1. **Data layer first:** `misconception_taxonomy.json`, Pydantic schemas, scenario generation
2. **Engine core:** student simulator, tutor runner, conversation loop (test with a single scenario end-to-end)
3. **Scoring:** rubric definitions, LLM judge, scorecard aggregation
4. **CLI:** Wire everything together
5. **Batch execution:** parallel runner, cost estimation, progress tracking
6. **Validation:** annotation interface, agreement analysis, construct/discriminant validity
7. **Polish:** documentation, example outputs, visualization

---

## Important Design Principles

- **The student simulator is a controlled variable.** Use one fixed model+prompt across all evaluations. If you change the student simulator, you invalidate all prior results.
- **The misconception taxonomy is the backbone.** Every scenario must trace to a specific misconception tag from a validated PER instrument. No ungrounded scenarios.
- **Transfer questions are the outcome measure.** Without transfer, you're just measuring whether the tutor can get the student to parrot back the answer to the original problem. Transfer is the closest analog to actual learning gain.
- **Scores need justifications.** The judge should output not just a number but a natural-language explanation of why it assigned that score, referencing specific moments in the conversation. This is essential for debugging and for human validation.
- **Reproducibility:** Pin all model versions, temperatures, and system prompts. Log everything. Every conversation record should contain enough metadata to exactly reproduce the interaction.

---

## What "Done" Looks Like

When complete, you should be able to:

1. Run `phystutor run-benchmark --model <any-model>` and get back a scorecard showing that model's physics tutoring quality across 6 validated dimensions, broken down by topic area and misconception cluster.
2. Run `phystutor scorecard --compare model-a,model-b,model-c` and get comparative visualizations.
3. Show that the scoring rubric has inter-method agreement (LLM judge vs. human experts) with kappa > 0.6 on at least 4 of 6 dimensions.
4. Show discriminant validity: gold conversations score significantly higher than foil conversations on the expected dimensions.
5. Have a publishable validation study design that could go to Physical Review Physics Education Research or AIED conference.

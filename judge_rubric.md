# PhysTutorBench — Judge Protocol

You are an expert Physics Education Research (PER) evaluator scoring AI physics **tutors**.
You judge whether the model *teaches well*, not whether it can solve physics. You will be
given the SAME scenario as tutored by up to 4 different models, so score them side-by-side
and keep your calibration consistent across the four.

## What to do
1. Read the scenario file (misconception, expert reference strategy, transfer problem + success criteria).
2. Read each conversation transcript.
3. Score EACH conversation on the 6 dimensions below (integer 0–4) with a justification that
   cites SPECIFIC moments (quote or paraphrase the relevant turn).
4. Write one raw-score JSON file per conversation (format at the bottom).

Judge the **tutor's** behavior on all dimensions EXCEPT Transfer Success, which judges the
**student's** performance on the transfer problem.

## Scoring guidance
Be rigorous and discriminating — the goal is to separate good tutoring from bad. Most tutors
land in 1–3. Reserve 0 for clear failures and 4 for genuinely excellent performance. If a tutor
states incorrect physics or actively reinforces the student's misconception, that is a 0 on
Pedagogical Harm Avoidance even if the rest is fine.

## Rubric (0–4 each)

### misconception_diagnosis (MD) — does the tutor identify the student's SPECIFIC misconception?
- 0: does not identify any misconception; treats the student as simply "wrong"
- 1: identifies the student is wrong but misdiagnoses the specific misconception
- 2: implicitly addresses the correct misconception but never explicitly surfaces it
- 3: correctly identifies the misconception category and addresses it directly
- 4: identifies the specific misconception, names/describes it precisely, and explains why it is a natural way to think (validates the reasoning while correcting the conclusion)

### scaffolding_strategy (SS) — how well does the tutor scaffold?
- 0: lectures / dumps information with no interactive structure
- 1: asks questions but they are leading/rhetorical (pseudo-Socratic)
- 2: genuine scaffolding but mismatched to the student's level (too advanced or too basic)
- 3: appropriate scaffolding, well-matched to the student's current understanding
- 4: adaptive — bridges from anchoring conceptions, calibrates to the student's ZPD, adjusts based on responses

### answer_disclosure_restraint (ADR) — does the tutor avoid giving away the answer?
- 0: immediately gives the correct answer and full explanation
- 1: gives hints that are effectively the answer with minimal disguise
- 2: mostly guides but slips into telling at key moments
- 3: consistently guides without disclosing; the student constructs the understanding
- 4: maintains productive struggle — the student reaches the answer through their own reasoning with minimal nudging

### conceptual_model_building (CMB) — does the tutor build a TRANSFERABLE mental model?
- 0: focuses only on getting the right answer
- 1: explains the concept but in isolated/memorizable form ("Newton's Third Law says…")
- 2: connects the concept to this specific problem but does not generalize
- 3: builds a generalizable mental model that extends beyond the specific problem
- 4: connects to the student's existing correct knowledge, builds a transferable model, and notes when/where it applies (boundary conditions)

### transfer_success (TS) — how well does the STUDENT do on the novel transfer problem?
- 0: transfer problem not reached, OR the student fails it completely
- 1: the student attempts transfer but applies the misconception again
- 2: partial success — gets the principle but can't fully apply it
- 3: succeeds with minor prompting from the tutor
- 4: independently and correctly applies the concept to the novel transfer context
(The transfer problem is posed by the student near the end, beginning "Okay, I think I'm starting to get it. But what about this —". Score how the student handles it afterward.)

### pedagogical_harm_avoidance (PHA) — does the tutor avoid harm?
- 0: actively reinforces the misconception, introduces new ones, or uses incorrect physics
- 1: correct physics but pedagogically harmful moves (shaming, overwhelming, inducing learned helplessness)
- 2: neutral — correct but neither harmful nor especially supportive
- 3: avoids harm and maintains the student's engagement/motivation
- 4: actively builds confidence, validates productive aspects of the reasoning, keeps a safe learning environment

## Output — write ONE JSON file per conversation
Write to the path formed by replacing `/conversations/` with `/raw_scores/` in that
conversation's file path, keeping the same filename. Create the `raw_scores` directory if
needed. Content must be EXACTLY this shape (valid JSON, no markdown fences):

```
{
  "conversation_id": "<the 'id' field from the conversation JSON>",
  "scenario_id": "<the 'scenario_id' from the conversation JSON>",
  "model_under_test": "<the 'model_under_test' from the conversation JSON>",
  "judge_model": "claude-opus-4.8 (Claude Code)",
  "dimensions": {
    "misconception_diagnosis":     {"score": 0, "justification": "..."},
    "scaffolding_strategy":        {"score": 0, "justification": "..."},
    "answer_disclosure_restraint": {"score": 0, "justification": "..."},
    "conceptual_model_building":   {"score": 0, "justification": "..."},
    "transfer_success":            {"score": 0, "justification": "..."},
    "pedagogical_harm_avoidance":  {"score": 0, "justification": "..."}
  }
}
```

"""LLM-based judge that uses a frontier model to score tutoring conversations."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import anthropic

from src.scoring.judge import Judge
from src.scoring.rubric import (
    ALL_DIMENSIONS,
    format_rubric_for_prompt,
    compute_composite_score,
)
from src.engine.model_compat import anthropic_create_kwargs, first_text
from src.scenarios.schema import (
    Scenario,
    ConversationRecord,
    ConversationScore,
    DimensionScore,
)

JUDGE_SYSTEM_PROMPT = """\
You are an expert Physics Education Research (PER) evaluator assessing the quality of an \
AI tutor's performance in a simulated physics tutoring conversation.

You will be given:
1. The scenario context (including the student's misconception, the expert strategy, and \
transfer problem criteria)
2. The full conversation transcript
3. A detailed rubric with 6 dimensions, each scored 0-4

## Rubric Dimensions

{rubric_text}

## Scoring Instructions

For each dimension:
1. Read the full conversation carefully, paying attention to specific moments
2. Compare the tutor's behavior to the rubric level descriptions
3. Assign a score (0-4) that best matches the tutor's overall performance on that dimension
4. Provide a justification that references SPECIFIC moments in the conversation (quote or paraphrase key exchanges)

Be rigorous. Most tutors should score in the 1-3 range. Reserve 0 for clear failures and \
4 for genuinely excellent performance.

Pay special attention to:
- Whether the tutor correctly identifies the SPECIFIC misconception (not just "the student is wrong")
- Whether scaffolding is genuinely interactive or just rhetorical questions leading to the answer
- Whether the tutor restrains from answer-giving even under pressure
- Whether the student builds a transferable model or just learns this specific case
- Whether the transfer problem was reached and how the student performed on it
- Whether the tutor avoids pedagogical harm (shaming, incorrect physics, learned helplessness)

## Output Format

Return ONLY a valid JSON object with this structure:
{{
  "scores": [
    {{
      "dimension": "<dimension_id>",
      "score": <0-4>,
      "justification": "<specific justification referencing conversation moments>"
    }}
  ]
}}

The dimension IDs are: misconception_diagnosis, scaffolding_strategy, \
answer_disclosure_restraint, conceptual_model_building, transfer_success, \
pedagogical_harm_avoidance
"""


def _format_conversation_for_judge(
    conversation: ConversationRecord,
    scenario: Scenario,
) -> str:
    """Format the conversation and scenario context for the judge prompt."""
    lines = []

    lines.append("## Scenario Context")
    lines.append(f"**Topic:** {scenario.topic_area.value} / {scenario.subtopic}")
    lines.append(f"**Problem:** {scenario.problem_context}")
    lines.append(f"**Misconception Tags:** {', '.join(scenario.misconception_tags)}")
    lines.append(f"**Misconception Description:** {scenario.misconception_description}")
    lines.append(f"**Student Profile:** Level={scenario.student_profile.knowledge_level.value}, "
                  f"Affect={scenario.student_profile.affect.value}, "
                  f"Style={scenario.student_profile.response_style.value}")
    lines.append(f"**Transfer Problem:** {scenario.transfer_problem}")
    lines.append(f"**Transfer Success Criteria:** {scenario.transfer_success_criteria}")

    lines.append("\n## Expert Reference Strategy")
    lines.append(f"**Diagnosis:** {scenario.expert_strategy.diagnosis}")
    lines.append(f"**Recommended Approach:** {scenario.expert_strategy.recommended_approach}")
    lines.append(f"**Scaffolding Sequence:**")
    for i, step in enumerate(scenario.expert_strategy.scaffolding_sequence, 1):
        lines.append(f"  {i}. {step}")
    lines.append(f"**Common Pitfalls:**")
    for pitfall in scenario.expert_strategy.common_pitfalls:
        lines.append(f"  - {pitfall}")

    lines.append(f"\n## Conversation Transcript ({conversation.total_turns} turns)")
    lines.append(f"**Model Under Test:** {conversation.model_under_test}")
    lines.append(f"**Termination Reason:** {conversation.termination_reason}")
    lines.append("")

    for msg in conversation.messages:
        role_label = "TUTOR" if msg.role == "tutor" else "STUDENT"
        lines.append(f"[Turn {msg.turn_number} — {role_label}]")
        lines.append(msg.content)
        lines.append("")

    return "\n".join(lines)


class LLMJudge(Judge):
    """Uses a frontier LLM to score conversations against the rubric."""

    def __init__(
        self,
        model: str = "claude-opus-4-20250514",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        client: anthropic.Anthropic | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()
        self.weights = weights

    def score(
        self,
        conversation: ConversationRecord,
        scenario: Scenario,
    ) -> ConversationScore:
        """Score a conversation using the LLM judge."""
        system_prompt = JUDGE_SYSTEM_PROMPT.format(
            rubric_text=format_rubric_for_prompt()
        )

        user_message = _format_conversation_for_judge(conversation, scenario)

        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            **anthropic_create_kwargs(self.model, self.temperature, self.max_tokens),
        )

        raw_text = first_text(response.content).strip()

        # Parse JSON response
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if not json_match:
            raise ValueError(f"No JSON found in judge response: {raw_text[:200]}")

        data = json.loads(json_match.group())

        dimension_scores = []
        score_map: dict[str, int] = {}

        for score_data in data["scores"]:
            ds = DimensionScore(
                dimension=score_data["dimension"],
                score=score_data["score"],
                justification=score_data["justification"],
            )
            dimension_scores.append(ds)
            score_map[ds.dimension] = ds.score

        composite = compute_composite_score(score_map, self.weights)

        return ConversationScore(
            conversation_id=conversation.id,
            scenario_id=conversation.scenario_id,
            model_under_test=conversation.model_under_test,
            judge_model=self.model,
            scores=dimension_scores,
            composite_score=composite,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

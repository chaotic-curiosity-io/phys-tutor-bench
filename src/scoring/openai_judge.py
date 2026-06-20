"""OpenAI-backed judge.

A second judge from a different model family than ``LLMJudge`` (Anthropic), so
the same conversation can be scored by two independent frontier judges and their
agreement reported (see ``compare_judges.py``). Reuses the exact rubric prompt
and parsing logic from ``llm_judge`` so the only thing that varies is the model.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import openai

from src.scoring.judge import Judge
from src.scoring.llm_judge import JUDGE_SYSTEM_PROMPT, _format_conversation_for_judge
from src.scoring.rubric import format_rubric_for_prompt, compute_composite_score
from src.engine.model_compat import openai_create_kwargs
from src.scenarios.schema import (
    Scenario,
    ConversationRecord,
    ConversationScore,
    DimensionScore,
)


class OpenAIJudge(Judge):
    """Uses an OpenAI frontier model to score conversations against the rubric.

    ``max_tokens`` defaults high because reasoning models (GPT-5.x) spend hidden
    reasoning tokens out of the same completion budget before emitting the JSON.
    """

    def __init__(
        self,
        model: str = "gpt-5.5",
        temperature: float = 0.0,
        max_tokens: int = 16000,
        client: openai.OpenAI | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client or openai.OpenAI()
        self.weights = weights

    def score(
        self,
        conversation: ConversationRecord,
        scenario: Scenario,
    ) -> ConversationScore:
        system_prompt = JUDGE_SYSTEM_PROMPT.format(
            rubric_text=format_rubric_for_prompt()
        )
        user_message = _format_conversation_for_judge(conversation, scenario)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            **openai_create_kwargs(self.model, self.temperature, self.max_tokens),
        )

        raw_text = (response.choices[0].message.content or "").strip()

        # Parse the first JSON object out of the response (same as the Anthropic judge).
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

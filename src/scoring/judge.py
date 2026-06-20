"""Base judge interface for scoring tutoring conversations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.scenarios.schema import Scenario, ConversationRecord, ConversationScore


class Judge(ABC):
    """Abstract base class for conversation judges."""

    @abstractmethod
    def score(
        self,
        conversation: ConversationRecord,
        scenario: Scenario,
    ) -> ConversationScore:
        """Score a conversation on all rubric dimensions.

        Args:
            conversation: The conversation record to evaluate.
            scenario: The scenario that generated this conversation (includes
                misconception tags, expert strategy, and transfer criteria).

        Returns:
            ConversationScore with scores and justifications for all dimensions.
        """
        ...


def create_judge(model: str, **kwargs) -> "Judge":
    """Factory: pick the judge backend by model-name prefix.

    OpenAI models (``gpt-*`` / o-series) route to ``OpenAIJudge``; everything
    else (Claude, default) routes to the Anthropic ``LLMJudge``. Mirrors
    ``create_tutor_backend`` so ``score --judge-model gpt-5.5`` just works.
    Imports are lazy to avoid a circular import with the judge implementations.
    """
    if model.startswith("gpt-") or model.startswith(("o1", "o3", "o4", "o5")):
        from src.scoring.openai_judge import OpenAIJudge

        return OpenAIJudge(model=model, **kwargs)
    from src.scoring.llm_judge import LLMJudge

    return LLMJudge(model=model, **kwargs)

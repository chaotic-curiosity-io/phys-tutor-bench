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

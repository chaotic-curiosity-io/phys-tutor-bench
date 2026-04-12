"""Stub for a reward-model-based judge (future implementation).

Once human annotations are available, this module will train a smaller model
to predict scores directly, replacing the more expensive LLM judge for
large-scale evaluations.
"""

from __future__ import annotations

from src.scoring.judge import Judge
from src.scenarios.schema import Scenario, ConversationRecord, ConversationScore


class RewardModelJudge(Judge):
    """Placeholder for a fine-tuned reward model judge.

    This will be implemented once sufficient human annotation data is
    collected through the validation framework. The training pipeline
    should:
    1. Use human-annotated conversation scores as training data
    2. Fine-tune a smaller model (e.g., Qwen2.5-1.5B) to predict
       scores on all 6 dimensions
    3. Validate against held-out human annotations
    4. Target kappa > 0.6 agreement with human judges
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        if model_path is None:
            raise NotImplementedError(
                "RewardModelJudge is not yet implemented. "
                "Collect human annotations first, then train the reward model. "
                "Use LLMJudge in the meantime."
            )

    def score(
        self,
        conversation: ConversationRecord,
        scenario: Scenario,
    ) -> ConversationScore:
        raise NotImplementedError(
            "RewardModelJudge.score() is not yet implemented."
        )

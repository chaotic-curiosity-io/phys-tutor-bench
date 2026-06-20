"""Tests for the OpenAI judge backend and the create_judge factory."""

import json
from unittest.mock import MagicMock

import pytest

from src.scenarios.schema import Scenario
from src.engine.student_simulator import StudentSimulator
from src.engine.conversation_loop import ConversationLoop
from src.scoring.openai_judge import OpenAIJudge
from src.scoring.llm_judge import LLMJudge
from src.scoring.judge import create_judge


@pytest.fixture
def sample_scenario():
    return Scenario(**{
        "id": "test-scenario-001",
        "topic_area": "mechanics",
        "subtopic": "newton_third_law",
        "level": "intro_algebra",
        "problem_context": "A truck hits a car. Compare forces.",
        "student_initial_response": "The truck hits harder because it's bigger.",
        "misconception_tags": ["AR1_greater_mass_greater_force"],
        "misconception_source": "FCI",
        "misconception_description": "Larger objects exert greater forces.",
        "common_prevalence": "high",
        "student_profile": {
            "knowledge_level": "intro_algebra",
            "knows": ["free_body_diagrams_basic"],
            "struggles_with": ["force_vs_effect"],
            "affect": "confident_but_wrong",
            "response_style": "verbose_reasoner",
        },
        "transfer_problem": "Astronaut pushes ISS. Compare forces.",
        "transfer_success_criteria": "Forces are equal and opposite.",
        "expert_strategy": {
            "diagnosis": "AR1 misconception",
            "recommended_approach": "bridging_analogy",
            "scaffolding_sequence": ["step 1", "step 2"],
            "common_pitfalls": ["pitfall 1"],
        },
    })


@pytest.fixture
def sample_record(sample_scenario):
    """Build a real ConversationRecord via the loop with mocked tutor + student."""
    mock_tutor = MagicMock()
    mock_tutor.model = "claude-opus-4-8"
    mock_tutor.respond.return_value = ("Let me help you reason about this.", 100)

    mock_client = MagicMock()
    resp = MagicMock()
    resp.content = [MagicMock(text="But the truck is bigger...")]
    resp.usage.input_tokens = 20
    resp.usage.output_tokens = 10
    mock_client.messages.create.return_value = resp

    student = StudentSimulator(scenario=sample_scenario, client=mock_client)
    loop = ConversationLoop(
        scenario=sample_scenario, tutor=mock_tutor, student=student, max_turns=2,
    )
    return loop.run()


def _judge_json():
    return json.dumps({
        "scores": [
            {"dimension": "misconception_diagnosis", "score": 3, "justification": "j"},
            {"dimension": "scaffolding_strategy", "score": 2, "justification": "j"},
            {"dimension": "answer_disclosure_restraint", "score": 1, "justification": "j"},
            {"dimension": "conceptual_model_building", "score": 3, "justification": "j"},
            {"dimension": "transfer_success", "score": 2, "justification": "j"},
            {"dimension": "pedagogical_harm_avoidance", "score": 4, "justification": "j"},
        ]
    })


def _mock_openai_client(content: str) -> MagicMock:
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=content))]
    client.chat.completions.create.return_value = completion
    return client


class TestOpenAIJudge:
    def test_score_parses_json(self, sample_record, sample_scenario):
        client = _mock_openai_client(_judge_json())
        judge = OpenAIJudge(model="gpt-5.5", client=client)
        result = judge.score(sample_record, sample_scenario)

        assert result.judge_model == "gpt-5.5"
        assert len(result.scores) == 6
        assert 0.0 <= result.composite_score <= 4.0
        assert result.model_under_test == "claude-opus-4-8"

    def test_reasoning_params_sent(self, sample_record, sample_scenario):
        """GPT-5.x judge must use max_completion_tokens and omit temperature."""
        client = _mock_openai_client(_judge_json())
        OpenAIJudge(model="gpt-5.5", client=client).score(sample_record, sample_scenario)

        call = client.chat.completions.create.call_args
        assert "max_completion_tokens" in call.kwargs
        assert "temperature" not in call.kwargs
        assert "max_tokens" not in call.kwargs

    def test_tolerates_prose_around_json(self, sample_record, sample_scenario):
        client = _mock_openai_client("Here is my evaluation:\n" + _judge_json() + "\nDone.")
        result = OpenAIJudge(model="gpt-5.5", client=client).score(sample_record, sample_scenario)
        assert len(result.scores) == 6


class TestCreateJudge:
    def test_openai_dispatch(self):
        assert isinstance(create_judge("gpt-5.5", client=MagicMock()), OpenAIJudge)
        assert isinstance(create_judge("o3", client=MagicMock()), OpenAIJudge)

    def test_anthropic_dispatch(self):
        assert isinstance(create_judge("claude-opus-4-8", client=MagicMock()), LLMJudge)

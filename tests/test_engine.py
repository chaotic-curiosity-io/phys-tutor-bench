"""Tests for the conversation engine subsystem."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.engine.student_simulator import StudentSimulator, build_student_system_prompt
from src.engine.tutor_runner import (
    AnthropicTutor,
    OpenAITutor,
    create_tutor_backend,
    DEFAULT_TUTOR_SYSTEM_PROMPT,
)
from src.engine.conversation_loop import ConversationLoop, save_conversation, load_conversation
from src.engine.batch_runner import load_scenarios, estimate_cost
from src.scenarios.schema import Scenario


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


class TestStudentSimulator:
    def test_system_prompt_construction(self, sample_scenario):
        """System prompt includes all scenario details."""
        prompt = build_student_system_prompt(sample_scenario)
        assert "Larger objects exert greater forces" in prompt
        assert "intro_algebra" in prompt
        assert "confident" in prompt.lower()
        assert "verbose" in prompt.lower()
        assert "truck" in prompt.lower() or "bigger" in prompt.lower()

    def test_simulator_init(self, sample_scenario):
        """StudentSimulator initializes with correct attributes."""
        sim = StudentSimulator(
            scenario=sample_scenario,
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            client=MagicMock(),
        )
        assert sim.model == "claude-sonnet-4-20250514"
        assert sim.temperature == 0.7

    def test_simulator_respond(self, sample_scenario):
        """StudentSimulator.respond calls the API correctly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="But the truck is heavier...")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response

        sim = StudentSimulator(
            scenario=sample_scenario,
            client=mock_client,
        )
        text, tokens = sim.respond([
            {"role": "user", "content": "What do you think about Newton's Third Law?"}
        ])

        assert text == "But the truck is heavier..."
        assert tokens == 150
        mock_client.messages.create.assert_called_once()

    def test_transfer_injection(self, sample_scenario):
        """Transfer injection modifies the system prompt."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="What about the astronaut?")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response

        sim = StudentSimulator(
            scenario=sample_scenario,
            client=mock_client,
        )
        sim.respond(
            [{"role": "user", "content": "Think about it..."}],
            inject_transfer=True,
        )

        # Check that the system prompt was modified to include transfer
        call_kwargs = mock_client.messages.create.call_args
        system = call_kwargs.kwargs.get("system", "") if call_kwargs.kwargs else ""
        assert "Transfer Problem" in system or "astronaut" in system.lower()


class TestTutorRunner:
    def test_create_anthropic_backend(self):
        """Creates Anthropic backend for claude models."""
        backend = create_tutor_backend("claude-sonnet-4-20250514")
        assert isinstance(backend, AnthropicTutor)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_create_openai_backend(self):
        """Creates OpenAI backend for gpt models."""
        backend = create_tutor_backend("gpt-4o")
        assert isinstance(backend, OpenAITutor)

    def test_anthropic_respond(self):
        """AnthropicTutor.respond calls the API correctly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Let's think about this...")]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 100
        mock_client.messages.create.return_value = mock_response

        tutor = AnthropicTutor(model="claude-sonnet-4-20250514", client=mock_client)
        text, tokens = tutor.respond(
            [{"role": "user", "content": "The truck hits harder."}],
            DEFAULT_TUTOR_SYSTEM_PROMPT,
        )

        assert text == "Let's think about this..."
        assert tokens == 300


class TestConversationLoop:
    def test_conversation_produces_record(self, sample_scenario):
        """ConversationLoop.run produces a valid ConversationRecord."""
        mock_tutor = MagicMock()
        mock_tutor.model = "test-model"
        mock_tutor.respond.return_value = ("Let me help you think about this.", 100)

        mock_student_client = MagicMock()
        mock_student_response = MagicMock()
        mock_student_response.content = [MagicMock(text="But the truck is bigger...")]
        mock_student_response.usage.input_tokens = 50
        mock_student_response.usage.output_tokens = 30
        mock_student_client.messages.create.return_value = mock_student_response

        student = StudentSimulator(
            scenario=sample_scenario,
            client=mock_student_client,
        )
        loop = ConversationLoop(
            scenario=sample_scenario,
            tutor=mock_tutor,
            student=student,
            max_turns=4,
        )

        record = loop.run()

        assert record.scenario_id == "test-scenario-001"
        assert record.model_under_test == "test-model"
        assert len(record.messages) > 0
        assert record.messages[0].role == "student"  # Initial response

    def test_save_and_load(self, sample_scenario, tmp_path):
        """Conversation records can be saved and loaded."""
        mock_tutor = MagicMock()
        mock_tutor.model = "test-model"
        mock_tutor.respond.return_value = ("Response.", 50)

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Student reply.")]
        mock_resp.usage.input_tokens = 20
        mock_resp.usage.output_tokens = 10
        mock_client.messages.create.return_value = mock_resp

        student = StudentSimulator(scenario=sample_scenario, client=mock_client)
        loop = ConversationLoop(
            scenario=sample_scenario,
            tutor=mock_tutor,
            student=student,
            max_turns=2,
        )
        record = loop.run()

        path = save_conversation(record, tmp_path)
        loaded = load_conversation(path)

        assert loaded.id == record.id
        assert loaded.scenario_id == record.scenario_id
        assert len(loaded.messages) == len(record.messages)


class TestDisengagementHeuristic:
    """The detector must catch real quit-signals without misfiring on filler words."""

    def _chk(self, text):
        import types
        fake = types.SimpleNamespace(
            _STRONG_DISENGAGEMENT=ConversationLoop._STRONG_DISENGAGEMENT,
            _AMBIGUOUS_DISENGAGEMENT=ConversationLoop._AMBIGUOUS_DISENGAGEMENT,
        )
        return ConversationLoop._check_disengagement(fake, text)

    @pytest.mark.parametrize("text", [
        "I give up.", "this is pointless", "I quit", "forget it",
        "Whatever.", "ugh, whatever.", "never mind",
    ])
    def test_genuine_disengagement_triggers(self, text):
        assert self._chk(text) is True

    @pytest.mark.parametrize("text", [
        "the fuel becomes exhaust and heat and whatever, but in space the toolbox keeps drifting",
        "a definite value along whatever axis we're about to measure, and it stays reproducible",
        "never mind that detail for now — what about the case where the bulb is brighter?",
    ])
    def test_filler_words_do_not_misfire(self, text):
        assert self._chk(text) is False


class TestBatchRunner:
    def test_estimate_cost(self):
        """Cost estimation produces reasonable numbers."""
        cost = estimate_cost(100)
        assert cost > 0
        assert cost < 100  # Sanity check

    def test_load_scenarios_empty(self, tmp_path):
        """Loading from empty directory returns empty list."""
        scenarios = load_scenarios(tmp_path)
        assert scenarios == []

    def test_load_scenarios(self, tmp_path, sample_scenario):
        """Can load scenarios from a directory."""
        (tmp_path / "test.json").write_text(
            sample_scenario.model_dump_json()
        )
        scenarios = load_scenarios(tmp_path)
        assert len(scenarios) == 1
        assert scenarios[0].id == "test-scenario-001"

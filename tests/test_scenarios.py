"""Tests for the scenario subsystem — taxonomy, schema, and generation."""

import json
from pathlib import Path

import pytest

from src.scenarios.schema import (
    Scenario,
    StudentProfile,
    ExpertStrategy,
    TopicArea,
    Level,
    Affect,
    ResponseStyle,
    Prevalence,
    MisconceptionEntry,
)
from src.scenarios.taxonomy import MisconceptionTaxonomy

TAXONOMY_PATH = Path("data/taxonomies/misconception_taxonomy.json")


# --- Fixtures ---

@pytest.fixture
def taxonomy():
    """Load the real taxonomy file."""
    return MisconceptionTaxonomy(TAXONOMY_PATH)


@pytest.fixture
def sample_scenario_data():
    """A valid scenario JSON dict for testing."""
    return {
        "id": "mech-n3l-ar-001",
        "topic_area": "mechanics",
        "subtopic": "newton_third_law",
        "level": "intro_algebra",
        "problem_context": "A 1000 kg car collides with a 100 kg motorcycle.",
        "student_initial_response": "The car exerts way more force because it's bigger.",
        "misconception_tags": ["AR1_greater_mass_greater_force"],
        "misconception_source": "FCI",
        "misconception_description": "Student believes larger objects exert greater forces.",
        "common_prevalence": "high",
        "student_profile": {
            "knowledge_level": "intro_algebra",
            "knows": ["free_body_diagrams_basic"],
            "struggles_with": ["distinguishing_force_from_effect"],
            "affect": "confident_but_wrong",
            "response_style": "verbose_reasoner",
        },
        "transfer_problem": "An astronaut pushes against the ISS. Compare the forces.",
        "transfer_success_criteria": "Student states forces are equal and opposite.",
        "expert_strategy": {
            "diagnosis": "AR1 — conflates force magnitude with effect",
            "recommended_approach": "bridging_analogy",
            "scaffolding_sequence": [
                "Acknowledge damage intuition",
                "Distinguish force from effect",
                "Use N2L to explain different accelerations",
            ],
            "common_pitfalls": [
                "Just stating N3L without addressing the intuition",
            ],
        },
    }


# --- Taxonomy Tests ---

class TestTaxonomy:
    def test_load_taxonomy(self, taxonomy):
        """Taxonomy loads without errors."""
        assert len(taxonomy.all_entries) > 0

    def test_taxonomy_has_all_topic_areas(self, taxonomy):
        """All four topic areas are represented."""
        topics = taxonomy.topic_summary()
        assert "mechanics" in topics
        assert "em" in topics
        assert "thermal_waves" in topics
        assert "modern_quantum" in topics

    def test_taxonomy_mechanics_count(self, taxonomy):
        """Mechanics should have the most misconceptions."""
        topics = taxonomy.topic_summary()
        assert topics["mechanics"] >= 15

    def test_get_by_id(self, taxonomy):
        """Can retrieve a specific misconception by ID."""
        entry = taxonomy.get("AR1_greater_mass_greater_force")
        assert entry is not None
        assert entry.topic_area == "mechanics"
        assert entry.subtopic == "newton_third_law"
        assert entry.source_instrument == "FCI"

    def test_get_nonexistent(self, taxonomy):
        """Returns None for nonexistent IDs."""
        assert taxonomy.get("FAKE_ID") is None

    def test_by_topic(self, taxonomy):
        """Filter by topic area works."""
        em_entries = taxonomy.by_topic("em")
        assert all(e.topic_area == "em" for e in em_entries)
        assert len(em_entries) > 0

    def test_by_subtopic(self, taxonomy):
        """Filter by subtopic works."""
        n3l = taxonomy.by_subtopic("newton_third_law")
        assert all(e.subtopic == "newton_third_law" for e in n3l)
        assert len(n3l) >= 3  # AR1, AR2, AR3

    def test_by_source(self, taxonomy):
        """Filter by source instrument works."""
        fci = taxonomy.by_source("FCI")
        assert all(e.source_instrument == "FCI" for e in fci)
        assert len(fci) > 0

    def test_by_prevalence(self, taxonomy):
        """Filter by prevalence works."""
        high = taxonomy.by_prevalence(Prevalence.HIGH)
        assert all(e.prevalence == Prevalence.HIGH for e in high)

    def test_subtopics(self, taxonomy):
        """Can get subtopics for a topic area."""
        mech_subtopics = taxonomy.get_subtopics("mechanics")
        assert "kinematics" in mech_subtopics
        assert "newton_third_law" in mech_subtopics

    def test_all_entries_valid(self, taxonomy):
        """All entries pass Pydantic validation."""
        for entry in taxonomy.all_entries:
            assert entry.id
            assert entry.description
            assert entry.topic_area
            assert entry.subtopic

    def test_unique_ids(self, taxonomy):
        """All misconception IDs are unique."""
        ids = [e.id for e in taxonomy.all_entries]
        assert len(ids) == len(set(ids))


# --- Schema Tests ---

class TestScenarioSchema:
    def test_valid_scenario(self, sample_scenario_data):
        """A valid scenario dict parses correctly."""
        scenario = Scenario(**sample_scenario_data)
        assert scenario.id == "mech-n3l-ar-001"
        assert scenario.topic_area == TopicArea.MECHANICS
        assert scenario.level == Level.INTRO_ALGEBRA

    def test_invalid_topic_area(self, sample_scenario_data):
        """Invalid topic area raises validation error."""
        sample_scenario_data["topic_area"] = "invalid_topic"
        with pytest.raises(Exception):
            Scenario(**sample_scenario_data)

    def test_student_profile_parsing(self, sample_scenario_data):
        """Student profile parses correctly."""
        scenario = Scenario(**sample_scenario_data)
        profile = scenario.student_profile
        assert profile.affect == Affect.CONFIDENT_BUT_WRONG
        assert profile.response_style == ResponseStyle.VERBOSE_REASONER

    def test_expert_strategy_parsing(self, sample_scenario_data):
        """Expert strategy parses correctly."""
        scenario = Scenario(**sample_scenario_data)
        strat = scenario.expert_strategy
        assert len(strat.scaffolding_sequence) > 0
        assert len(strat.common_pitfalls) > 0

    def test_scenario_roundtrip(self, sample_scenario_data):
        """Scenario can be serialized and deserialized."""
        scenario = Scenario(**sample_scenario_data)
        data = json.loads(scenario.model_dump_json())
        reconstructed = Scenario(**data)
        assert reconstructed.id == scenario.id
        assert reconstructed.misconception_tags == scenario.misconception_tags

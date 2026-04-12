"""Tests for the scoring subsystem — rubric, judge, scorecard."""

import json

import pytest

from src.scoring.rubric import (
    ALL_DIMENSIONS,
    DIMENSION_BY_ID,
    DIMENSION_BY_ABBREV,
    DEFAULT_WEIGHTS,
    compute_composite_score,
    format_rubric_for_prompt,
    MISCONCEPTION_DIAGNOSIS,
)
from src.scoring.scorecard import Scorecard
from src.scenarios.schema import ConversationScore, DimensionScore


# --- Fixtures ---

@pytest.fixture
def sample_scores():
    """Create sample ConversationScore objects for testing."""
    def make_score(conv_id, model, dim_scores, composite):
        return ConversationScore(
            conversation_id=conv_id,
            scenario_id=f"scenario-{conv_id}",
            model_under_test=model,
            judge_model="claude-opus-4-20250514",
            scores=[
                DimensionScore(dimension=dim, score=s, justification=f"Test justification for {dim}")
                for dim, s in dim_scores.items()
            ],
            composite_score=composite,
            timestamp="2025-01-15T00:00:00Z",
        )

    return [
        make_score("conv-001", "model-a", {
            "misconception_diagnosis": 3,
            "scaffolding_strategy": 3,
            "answer_disclosure_restraint": 2,
            "conceptual_model_building": 3,
            "transfer_success": 2,
            "pedagogical_harm_avoidance": 4,
        }, 2.85),
        make_score("conv-002", "model-a", {
            "misconception_diagnosis": 2,
            "scaffolding_strategy": 2,
            "answer_disclosure_restraint": 3,
            "conceptual_model_building": 2,
            "transfer_success": 1,
            "pedagogical_harm_avoidance": 3,
        }, 2.15),
        make_score("conv-003", "model-b", {
            "misconception_diagnosis": 4,
            "scaffolding_strategy": 4,
            "answer_disclosure_restraint": 4,
            "conceptual_model_building": 4,
            "transfer_success": 3,
            "pedagogical_harm_avoidance": 4,
        }, 3.85),
        make_score("conv-004", "model-b", {
            "misconception_diagnosis": 1,
            "scaffolding_strategy": 1,
            "answer_disclosure_restraint": 0,
            "conceptual_model_building": 1,
            "transfer_success": 0,
            "pedagogical_harm_avoidance": 2,
        }, 0.85),
    ]


# --- Rubric Tests ---

class TestRubric:
    def test_all_dimensions_defined(self):
        """All 6 dimensions are defined."""
        assert len(ALL_DIMENSIONS) == 6

    def test_dimension_ids_unique(self):
        """Dimension IDs are unique."""
        ids = [d.id for d in ALL_DIMENSIONS]
        assert len(ids) == len(set(ids))

    def test_dimension_levels(self):
        """Each dimension has exactly 5 levels (0-4)."""
        for dim in ALL_DIMENSIONS:
            assert len(dim.levels) == 5
            scores = [l.score for l in dim.levels]
            assert scores == [0, 1, 2, 3, 4]

    def test_get_level(self):
        """Can retrieve a specific score level."""
        level = MISCONCEPTION_DIAGNOSIS.get_level(3)
        assert level.score == 3
        assert "correctly identifies" in level.description.lower()

    def test_get_invalid_level(self):
        """Invalid score raises ValueError."""
        with pytest.raises(ValueError):
            MISCONCEPTION_DIAGNOSIS.get_level(5)

    def test_dimension_by_id(self):
        """Can look up dimensions by ID."""
        assert DIMENSION_BY_ID["misconception_diagnosis"] == MISCONCEPTION_DIAGNOSIS

    def test_dimension_by_abbrev(self):
        """Can look up dimensions by abbreviation."""
        assert DIMENSION_BY_ABBREV["MD"] == MISCONCEPTION_DIAGNOSIS

    def test_weights_sum_to_one(self):
        """Default weights sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_weights_cover_all_dimensions(self):
        """Weights cover all dimension IDs."""
        for dim in ALL_DIMENSIONS:
            assert dim.id in DEFAULT_WEIGHTS

    def test_composite_score(self):
        """Composite score computation is correct."""
        scores = {
            "misconception_diagnosis": 4,
            "scaffolding_strategy": 4,
            "answer_disclosure_restraint": 4,
            "conceptual_model_building": 4,
            "transfer_success": 4,
            "pedagogical_harm_avoidance": 4,
        }
        assert compute_composite_score(scores) == 4.0

    def test_composite_score_zeros(self):
        """All zeros give zero composite."""
        scores = {dim.id: 0 for dim in ALL_DIMENSIONS}
        assert compute_composite_score(scores) == 0.0

    def test_format_rubric_for_prompt(self):
        """Rubric formatting produces readable text."""
        text = format_rubric_for_prompt()
        assert "Misconception Diagnosis" in text
        assert "Transfer Success" in text
        assert "0" in text
        assert "4" in text


# --- Scorecard Tests ---

class TestScorecard:
    def test_models(self, sample_scores):
        """Scorecard identifies all models."""
        sc = Scorecard(sample_scores)
        assert set(sc.models) == {"model-a", "model-b"}

    def test_model_scores(self, sample_scores):
        """Can retrieve scores for a specific model."""
        sc = Scorecard(sample_scores)
        a_scores = sc.model_scores("model-a")
        assert len(a_scores) == 2

    def test_dimension_average(self, sample_scores):
        """Dimension averages are computed correctly."""
        sc = Scorecard(sample_scores)
        avg = sc.dimension_average("model-a", "misconception_diagnosis")
        assert avg == 2.5  # (3 + 2) / 2

    def test_composite_average(self, sample_scores):
        """Composite averages are computed correctly."""
        sc = Scorecard(sample_scores)
        avg = sc.composite_average("model-a")
        assert abs(avg - 2.5) < 0.01  # (2.85 + 2.15) / 2

    def test_to_json(self, sample_scores):
        """Scorecard exports to valid JSON."""
        sc = Scorecard(sample_scores)
        data = sc.to_json()
        assert "model-a" in data
        assert "model-b" in data
        assert "composite_mean" in data["model-a"]
        assert "dimensions" in data["model-a"]

    def test_generate_comparison_plot(self, sample_scores, tmp_path):
        """Comparison plot is generated without errors."""
        sc = Scorecard(sample_scores)
        path = sc.generate_comparison_plot(tmp_path / "comparison.png")
        assert path.exists()

    def test_generate_heatmap(self, sample_scores, tmp_path):
        """Heatmap is generated without errors."""
        sc = Scorecard(sample_scores)
        path = sc.generate_heatmap(tmp_path / "heatmap.png")
        assert path.exists()

    def test_generate_distribution_plots(self, sample_scores, tmp_path):
        """Distribution plots are generated for all dimensions."""
        sc = Scorecard(sample_scores)
        paths = sc.generate_distribution_plots(tmp_path / "dists")
        assert len(paths) == 6
        assert all(p.exists() for p in paths)

    def test_per_topic_breakdown(self, sample_scores):
        """Topic breakdown infers topics from scenario IDs."""
        sc = Scorecard(sample_scores)
        # Our sample scenario IDs don't have valid prefixes, so they'll be "unknown"
        breakdown = sc.per_topic_breakdown("model-a")
        assert len(breakdown) > 0

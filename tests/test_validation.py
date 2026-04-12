"""Tests for the validation framework — agreement analysis, construct validity."""

import sqlite3
from pathlib import Path

import pytest
import numpy as np

from src.validation.agreement_analysis import (
    cohens_kappa,
    weighted_cohens_kappa,
    krippendorff_alpha,
    spearman_rho,
    confusion_matrix,
)
from src.validation.construct_validity import (
    pearson_correlation,
    correlation_matrix,
)
from src.validation.human_annotation_interface import init_db, save_annotation, get_annotations
from src.scenarios.schema import ConversationScore, DimensionScore


# --- Agreement Analysis Tests ---

class TestCohensKappa:
    def test_perfect_agreement(self):
        """Perfect agreement should give kappa = 1."""
        r1 = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        r2 = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        assert cohens_kappa(r1, r2) == 1.0

    def test_no_agreement(self):
        """Random/no agreement should give kappa near 0."""
        r1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        r2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        kappa = cohens_kappa(r1, r2)
        assert kappa < 0.1

    def test_partial_agreement(self):
        """Partial agreement gives kappa between 0 and 1."""
        r1 = [0, 1, 2, 3, 4, 2, 3, 1, 2, 3]
        r2 = [0, 1, 2, 3, 4, 3, 2, 1, 3, 2]
        kappa = cohens_kappa(r1, r2)
        assert 0 < kappa < 1

    def test_empty_input(self):
        """Empty input returns 0."""
        assert cohens_kappa([], []) == 0.0


class TestWeightedKappa:
    def test_perfect_agreement(self):
        """Perfect agreement gives weighted kappa = 1."""
        r1 = [0, 1, 2, 3, 4]
        r2 = [0, 1, 2, 3, 4]
        assert weighted_cohens_kappa(r1, r2) == 1.0

    def test_adjacent_disagreement(self):
        """Adjacent disagreement is penalized less than distant."""
        r1 = [0, 1, 2, 3, 4]
        r2_near = [1, 2, 3, 4, 3]  # Off by 1
        r2_far = [4, 3, 0, 1, 0]   # Off by many

        kappa_near = weighted_cohens_kappa(r1, r2_near)
        kappa_far = weighted_cohens_kappa(r1, r2_far)
        assert kappa_near > kappa_far


class TestKrippendorffAlpha:
    def test_perfect_agreement(self):
        """Perfect agreement among all annotators."""
        data = [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        ]
        alpha = krippendorff_alpha(data)
        assert alpha == 1.0

    def test_with_missing_data(self):
        """Alpha handles missing ratings."""
        data = [
            [0, 1, 2, None, 4],
            [0, 1, None, 3, 4],
            [None, 1, 2, 3, 4],
        ]
        alpha = krippendorff_alpha(data)
        assert alpha > 0  # Should still compute with missing data


class TestSpearmanRho:
    def test_perfect_correlation(self):
        """Perfectly ordered data gives rho = 1."""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        assert abs(spearman_rho(x, y) - 1.0) < 0.001

    def test_perfect_inverse(self):
        """Perfectly inversely ordered gives rho = -1."""
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        assert abs(spearman_rho(x, y) - (-1.0)) < 0.001

    def test_no_correlation(self):
        """Random data gives rho near 0."""
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 2, 5]
        rho = spearman_rho(x, y)
        assert abs(rho) < 0.8  # Not strongly correlated


class TestConfusionMatrix:
    def test_shape(self):
        """Matrix has correct shape."""
        r1 = [0, 1, 2, 3, 4]
        r2 = [0, 1, 2, 3, 4]
        cm = confusion_matrix(r1, r2)
        assert cm.shape == (5, 5)

    def test_diagonal(self):
        """Perfect agreement puts all values on diagonal."""
        r1 = [0, 1, 2, 3, 4]
        r2 = [0, 1, 2, 3, 4]
        cm = confusion_matrix(r1, r2)
        assert np.sum(np.diag(cm)) == 5


# --- Construct Validity Tests ---

class TestPearsonCorrelation:
    def test_perfect_positive(self):
        """Perfect positive correlation."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert abs(pearson_correlation(x, y) - 1.0) < 0.001

    def test_perfect_negative(self):
        """Perfect negative correlation."""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        assert abs(pearson_correlation(x, y) - (-1.0)) < 0.001

    def test_no_correlation(self):
        """Uncorrelated data."""
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 2, 5]
        r = pearson_correlation(x, y)
        assert abs(r) < 0.8

    def test_constant_input(self):
        """Constant input returns 0."""
        x = [3, 3, 3, 3]
        y = [1, 2, 3, 4]
        assert pearson_correlation(x, y) == 0.0


# --- Annotation Database Tests ---

class TestAnnotationDB:
    def test_init_db(self, tmp_path):
        """Database initializes correctly."""
        conn = init_db(tmp_path / "test.db")
        assert conn is not None
        conn.close()

    def test_save_and_retrieve(self, tmp_path):
        """Can save and retrieve annotations."""
        conn = init_db(tmp_path / "test.db")
        save_annotation(
            conn,
            annotator_id="ann1",
            conversation_id="conv1",
            scenario_id="sc1",
            dimension="misconception_diagnosis",
            score=3,
            justification="Good identification",
        )

        results = get_annotations(conn, "conv1")
        assert len(results) == 1
        assert results[0]["score"] == 3
        assert results[0]["annotator_id"] == "ann1"
        conn.close()

    def test_upsert(self, tmp_path):
        """Saving again for same annotator/conversation/dimension updates the score."""
        conn = init_db(tmp_path / "test.db")
        save_annotation(conn, "ann1", "conv1", "sc1", "misconception_diagnosis", 2, "First")
        save_annotation(conn, "ann1", "conv1", "sc1", "misconception_diagnosis", 4, "Updated")

        results = get_annotations(conn, "conv1")
        assert len(results) == 1
        assert results[0]["score"] == 4
        conn.close()

    def test_multiple_annotators(self, tmp_path):
        """Multiple annotators can annotate the same conversation."""
        conn = init_db(tmp_path / "test.db")
        save_annotation(conn, "ann1", "conv1", "sc1", "misconception_diagnosis", 3, "A")
        save_annotation(conn, "ann2", "conv1", "sc1", "misconception_diagnosis", 2, "B")

        results = get_annotations(conn, "conv1")
        assert len(results) == 2
        conn.close()

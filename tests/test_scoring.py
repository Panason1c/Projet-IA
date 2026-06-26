"""
test_scoring.py — Tests unitaires pour src/scoring.py
"""
import pytest
from src.scoring import score_global, interpreter_score, resume_scores


class TestScoreGlobal:
    """Tests for the weighted coverage formula."""

    def test_score_global_equal_weights(self):
        """Test case from specifications: blocks 0.85, 0.78, 0.40 with equal weights."""
        scores_blocs = {1: 0.85, 2: 0.78, 3: 0.40}
        poids = {1: 1.0, 2: 1.0, 3: 1.0}

        result = score_global(scores_blocs, poids)
        # Expected: (0.85 + 0.78 + 0.40) / 3 = 2.03 / 3 ≈ 0.6767
        expected = (0.85 + 0.78 + 0.40) / 3
        assert result == pytest.approx(expected, abs=0.01)

    def test_score_global_weighted(self):
        """Test with different weights per block."""
        scores_blocs = {1: 0.8, 2: 0.6, 3: 0.4}
        poids = {1: 2.0, 2: 1.0, 3: 1.0}

        result = score_global(scores_blocs, poids)
        # Expected: (2*0.8 + 1*0.6 + 1*0.4) / (2 + 1 + 1) = 2.8 / 4 = 0.7
        expected = (2 * 0.8 + 1 * 0.6 + 1 * 0.4) / 4
        assert result == pytest.approx(expected, abs=0.01)

    def test_score_global_empty_dict(self):
        """Test with empty scores dictionary."""
        result = score_global({})
        assert result == 0.0

    def test_score_global_zero_weights(self):
        """Test with zero total weights (edge case)."""
        scores_blocs = {1: 0.5, 2: 0.6}
        poids = {1: 0.0, 2: 0.0}

        result = score_global(scores_blocs, poids)
        assert result == 0.0

    def test_score_global_default_weights(self):
        """Test using default weights from config."""
        scores_blocs = {1: 0.7, 2: 0.7, 3: 0.7, 4: 0.7, 5: 0.7, 6: 0.7}
        # All weights are 1.0 by default
        result = score_global(scores_blocs)
        expected = 0.7
        assert result == pytest.approx(expected, abs=0.01)


class TestInterpreterScore:
    """Tests for score interpretation based on thresholds."""

    def test_bon_score(self):
        """Test interpretation for score >= 0.7 (SEUIL_BON)."""
        level, description = interpreter_score(0.75)
        assert level == "bon"
        assert "Bonne" in description or "bonne" in description

    def test_moyen_score(self):
        """Test interpretation for 0.5 <= score < 0.7."""
        level, description = interpreter_score(0.60)
        assert level == "moyen"
        assert "partielle" in description or "Partielle" in description

    def test_faible_score(self):
        """Test interpretation for score < 0.5."""
        level, description = interpreter_score(0.40)
        assert level == "faible"
        assert "insuffisante" in description or "Insuffisante" in description

    def test_exact_seuil_bon(self):
        """Test interpretation at exact threshold 0.7."""
        level, description = interpreter_score(0.7)
        assert level == "bon"

    def test_exact_seuil_moyen(self):
        """Test interpretation at exact threshold 0.5."""
        level, description = interpreter_score(0.5)
        assert level == "moyen"

    def test_score_zero(self):
        """Test interpretation for zero score."""
        level, description = interpreter_score(0.0)
        assert level == "faible"

    def test_score_one(self):
        """Test interpretation for perfect score."""
        level, description = interpreter_score(1.0)
        assert level == "bon"


class TestResumeScores:
    """Tests for comprehensive score summary."""

    def test_resume_scores_structure(self):
        """Test that resume_scores returns correct structure."""
        scores_blocs = {1: 0.85, 2: 0.78}
        noms_blocs = {1: "Data Analysis", 2: "Machine Learning"}

        result = resume_scores(scores_blocs, noms_blocs)

        # Check top-level keys
        assert "score_global" in result
        assert "niveau_global" in result
        assert "description_globale" in result
        assert "blocs" in result

        # Check blocs array
        assert isinstance(result["blocs"], list)
        assert len(result["blocs"]) == 2

        # Check first bloc structure
        bloc = result["blocs"][0]
        assert "id" in bloc
        assert "nom" in bloc
        assert "score" in bloc
        assert "poids" in bloc
        assert "niveau" in bloc
        assert "description" in bloc

    def test_resume_scores_global_score(self):
        """Test that global score is correctly computed."""
        scores_blocs = {1: 0.8, 2: 0.6}
        noms_blocs = {1: "Block 1", 2: "Block 2"}

        result = resume_scores(scores_blocs, noms_blocs)
        expected_global = (0.8 + 0.6) / 2
        assert result["score_global"] == pytest.approx(expected_global, abs=0.01)

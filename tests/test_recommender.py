"""
test_recommender.py — Tests unitaires pour src/recommender.py
"""
import pytest
from src.recommender import (
    calculer_score_metier,
    top_n_metiers,
    analyser_ecarts,
    construire_contexte_rag
)


class TestCalculerScoreMetier:
    """Tests for job score calculation."""

    def test_calculer_score_metier_simple(self):
        """Test score calculation with simple competency scores."""
        metier_info = {
            "title": "Test Job",
            "competency_ids": ["C01", "C02", "C03"]
        }
        scores_competences = {"C01": 0.9, "C02": 0.8, "C03": 0.7}

        result = calculer_score_metier(metier_info, scores_competences)
        expected = (0.9 + 0.8 + 0.7) / 3
        assert result == pytest.approx(expected, abs=0.01)

    def test_calculer_score_metier_missing_competencies(self):
        """Test score when some competencies have no score (default 0.0)."""
        metier_info = {
            "title": "Test Job",
            "competency_ids": ["C01", "C02", "C99"]
        }
        scores_competences = {"C01": 0.8, "C02": 0.6}

        result = calculer_score_metier(metier_info, scores_competences)
        expected = (0.8 + 0.6 + 0.0) / 3
        assert result == pytest.approx(expected, abs=0.01)

    def test_calculer_score_metier_no_competencies(self):
        """Test score when job has no required competencies."""
        metier_info = {
            "title": "Test Job",
            "competency_ids": []
        }
        scores_competences = {"C01": 0.8}

        result = calculer_score_metier(metier_info, scores_competences)
        assert result == 0.0


class TestTopNMetiers:
    """Tests for top-N job recommendation."""

    def test_top_n_metiers_count(self):
        """Test that top_n_metiers returns exactly N results."""
        metiers = {
            "J01": {"title": "Job 1", "competency_ids": ["C01", "C02"]},
            "J02": {"title": "Job 2", "competency_ids": ["C03", "C04"]},
            "J03": {"title": "Job 3", "competency_ids": ["C05", "C06"]},
            "J04": {"title": "Job 4", "competency_ids": ["C07", "C08"]},
        }
        scores_competences = {
            "C01": 0.9, "C02": 0.8,  # J01: avg 0.85
            "C03": 0.7, "C04": 0.6,  # J02: avg 0.65
            "C05": 0.5, "C06": 0.4,  # J03: avg 0.45
            "C07": 0.95, "C08": 0.9,  # J04: avg 0.925
        }

        result = top_n_metiers(metiers, scores_competences, n=3)
        assert len(result) == 3

    def test_top_n_metiers_sorted_descending(self):
        """Test that results are sorted by score in descending order."""
        metiers = {
            "J01": {"title": "Job 1", "competency_ids": ["C01"]},
            "J02": {"title": "Job 2", "competency_ids": ["C02"]},
            "J03": {"title": "Job 3", "competency_ids": ["C03"]},
        }
        scores_competences = {
            "C01": 0.5,
            "C02": 0.9,
            "C03": 0.7,
        }

        result = top_n_metiers(metiers, scores_competences, n=3)

        # Check descending order
        for i in range(len(result) - 1):
            assert result[i]["score"] >= result[i + 1]["score"]

    def test_top_n_metiers_default_n(self):
        """Test that default N=3 is used."""
        metiers = {
            f"J{i:02d}": {"title": f"Job {i}", "competency_ids": ["C01"]}
            for i in range(1, 6)
        }
        scores_competences = {
            "C01": 0.5 + i * 0.1 for i in range(5)
        }

        result = top_n_metiers(metiers, scores_competences)
        assert len(result) == 3

    def test_top_n_metiers_structure(self):
        """Test that each result has required fields."""
        metiers = {
            "J01": {"title": "Job 1", "competency_ids": ["C01", "C02"]},
        }
        scores_competences = {"C01": 0.8, "C02": 0.7}

        result = top_n_metiers(metiers, scores_competences, n=1)
        job = result[0]

        assert "job_id" in job
        assert "title" in job
        assert "score" in job
        assert "competency_ids" in job
        assert "competency_scores" in job

    def test_top_n_metiers_no_duplicates(self):
        """Test that top-3 results don't contain duplicates."""
        metiers = {
            f"J{i:02d}": {"title": f"Job {i}", "competency_ids": ["C01"]}
            for i in range(1, 6)
        }
        scores_competences = {"C01": 0.5}

        result = top_n_metiers(metiers, scores_competences, n=3)
        job_ids = [job["job_id"] for job in result]
        assert len(job_ids) == len(set(job_ids))


class TestAnalyserEcarts:
    """Tests for gap analysis (weak competencies)."""

    def test_analyser_ecarts_sorted_ascending(self):
        """Test that results are sorted by score in ascending order."""
        scores_competences = {
            "C01": 0.9,
            "C02": 0.5,
            "C03": 0.7,
            "C04": 0.3,
        }
        competency_texts = {
            "C01": "High score",
            "C02": "Mid score",
            "C03": "Higher mid",
            "C04": "Low score",
        }

        result = analyser_ecarts(scores_competences, competency_texts, n_faibles=4)

        # Check ascending order
        for i in range(len(result) - 1):
            assert result[i]["score"] <= result[i + 1]["score"]

    def test_analyser_ecarts_count(self):
        """Test that exactly n_faibles results are returned."""
        scores_competences = {
            f"C{i:02d}": 0.5 for i in range(1, 11)
        }
        competency_texts = {
            f"C{i:02d}": f"Competency {i}" for i in range(1, 11)
        }

        result = analyser_ecarts(scores_competences, competency_texts, n_faibles=5)
        assert len(result) == 5

    def test_analyser_ecarts_structure(self):
        """Test that each result has required fields."""
        scores_competences = {"C01": 0.5, "C02": 0.7}
        competency_texts = {"C01": "Comp 1", "C02": "Comp 2"}

        result = analyser_ecarts(scores_competences, competency_texts, n_faibles=2)

        for comp in result:
            assert "id" in comp
            assert "texte" in comp
            assert "score" in comp

    def test_analyser_ecarts_missing_text(self):
        """Test that missing text defaults to competency ID."""
        scores_competences = {"C01": 0.5}
        competency_texts = {}

        result = analyser_ecarts(scores_competences, competency_texts, n_faibles=1)
        assert result[0]["texte"] == "C01"


class TestConstructireContexteRAG:
    """Tests for RAG context building."""

    def test_construire_contexte_rag_returns_string(self):
        """Test that function returns a string."""
        resume_scores = {
            "score_global": 0.68,
            "niveau_global": "moyen",
            "description_globale": "Couverture partielle",
            "blocs": [
                {"nom": "Data Analysis", "score": 0.85, "description": "Bonne"}
            ]
        }
        top_metiers = [
            {"title": "Data Scientist", "score": 0.75}
        ]
        competences_faibles = [
            {"texte": "NLP", "score": 0.2}
        ]

        result = construire_contexte_rag(resume_scores, top_metiers, competences_faibles)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_construire_contexte_rag_contains_key_info(self):
        """Test that context includes required information."""
        resume_scores = {
            "score_global": 0.68,
            "niveau_global": "moyen",
            "description_globale": "Couverture partielle",
            "blocs": []
        }
        top_metiers = [
            {"title": "Data Scientist", "score": 0.75}
        ]
        competences_faibles = []

        result = construire_contexte_rag(resume_scores, top_metiers, competences_faibles)

        # Check for key content
        assert "couverture" in result.lower()
        assert "Data Scientist" in result

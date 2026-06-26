"""
test_referentiel.py — Tests unitaires pour src/referentiel.py
"""
import pytest
from src.referentiel import (
    charger_competences,
    charger_metiers,
    get_blocs,
    get_metiers,
    valider_integrite
)


class TestChargerCompetences:
    """Tests for loading competencies from CSV."""

    def test_charger_competences_structure(self):
        """Test that competences CSV is loaded with expected columns."""
        df = charger_competences()
        expected_cols = {"CompetencyID", "Competency", "BlockID", "BlockName"}
        assert expected_cols.issubset(set(df.columns))

    def test_charger_competences_count(self):
        """Test that competences CSV contains 36 competencies."""
        df = charger_competences()
        # According to specs, 36 competencies (C01 to C36)
        assert len(df) == 36

    def test_charger_competences_no_duplicates(self):
        """Test that CompetencyID values are unique."""
        df = charger_competences()
        assert not df["CompetencyID"].duplicated().any()

    def test_charger_competences_blockid_type(self):
        """Test that BlockID is converted to integer."""
        df = charger_competences()
        assert df["BlockID"].dtype == "int64" or df["BlockID"].dtype == "int32"


class TestChargerMetiers:
    """Tests for loading jobs from CSV."""

    def test_charger_metiers_structure(self):
        """Test that metiers CSV is loaded with expected columns."""
        df = charger_metiers()
        expected_cols = {"JobID", "JobTitle", "RequiredCompetencies"}
        assert expected_cols.issubset(set(df.columns))

    def test_charger_metiers_count(self):
        """Test that metiers CSV contains 8 jobs."""
        df = charger_metiers()
        # According to specs, 8 jobs (J01 to J08)
        assert len(df) == 8

    def test_charger_metiers_no_duplicates(self):
        """Test that JobID values are unique."""
        df = charger_metiers()
        assert not df["JobID"].duplicated().any()


class TestGetBlocs:
    """Tests for building blocks dictionary."""

    def test_get_blocs_structure(self):
        """Test that get_blocs returns correct dictionary structure."""
        blocs = get_blocs()

        # Should have 6 blocks (1-6)
        assert len(blocs) == 6

        # Each block should have 'name' and 'competencies'
        for block_id, bloc_info in blocs.items():
            assert "name" in bloc_info
            assert "competencies" in bloc_info
            assert isinstance(bloc_info["competencies"], list)

    def test_get_blocs_competency_structure(self):
        """Test that each competency has id and text."""
        blocs = get_blocs()

        for block_id, bloc_info in blocs.items():
            for comp in bloc_info["competencies"]:
                assert "id" in comp
                assert "text" in comp
                assert isinstance(comp["id"], str)
                assert isinstance(comp["text"], str)


class TestGetMetiers:
    """Tests for building jobs dictionary."""

    def test_get_metiers_structure(self):
        """Test that get_metiers returns correct dictionary structure."""
        metiers = get_metiers()

        # Should have 8 jobs
        assert len(metiers) == 8

        # Each job should have required fields
        for job_id, job_info in metiers.items():
            assert "title" in job_info
            assert "competency_ids" in job_info
            assert "competency_texts" in job_info

    def test_get_metiers_competency_parsing(self):
        """Test that required competencies are correctly parsed."""
        metiers = get_metiers()

        for job_id, job_info in metiers.items():
            comp_ids = job_info["competency_ids"]
            comp_texts = job_info["competency_texts"]

            # Both lists should have same length
            assert len(comp_ids) == len(comp_texts)

            # All IDs should start with 'C'
            for cid in comp_ids:
                assert cid.startswith("C")

    def test_get_metiers_tolerant_parsing(self):
        """Test that semicolon parsing handles spaces."""
        metiers = get_metiers()

        # All parsed IDs should be clean (no leading/trailing spaces)
        for job_id, job_info in metiers.items():
            for cid in job_info["competency_ids"]:
                assert cid == cid.strip()


class TestValiderIntegrite:
    """Tests for integrity validation."""

    def test_valider_integrite_no_anomalies(self):
        """Test that validation returns empty list for correct data."""
        anomalies = valider_integrite()
        assert isinstance(anomalies, list)
        assert len(anomalies) == 0, f"Unexpected anomalies: {anomalies}"

    def test_valider_integrite_all_competencies_exist(self):
        """Test that all required competencies exist in competences.csv."""
        df_comp = charger_competences()
        df_metiers = charger_metiers()

        valid_ids = set(df_comp["CompetencyID"])

        for _, row in df_metiers.iterrows():
            comp_ids = [cid.strip() for cid in row["RequiredCompetencies"].split(";") if cid.strip()]
            for cid in comp_ids:
                assert cid in valid_ids, f"CompetencyID {cid} not found in competences"

    def test_valider_integrite_unique_ids(self):
        """Test that IDs are unique within each file."""
        df_comp = charger_competences()
        df_metiers = charger_metiers()

        # No duplicate CompetencyIDs
        assert not df_comp["CompetencyID"].duplicated().any()

        # No duplicate JobIDs
        assert not df_metiers["JobID"].duplicated().any()

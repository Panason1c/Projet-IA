"""
test_cache.py — Tests unitaires pour src/cache.py
"""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from src.cache import (
    lire_cache,
    ecrire_cache,
    appel_avec_cache,
    vider_cache,
    taille_cache,
    _hash_prompt
)


class TestHashPrompt:
    """Tests for prompt hashing."""

    def test_hash_prompt_deterministic(self):
        """Test that hashing the same prompt always produces the same hash."""
        prompt = "Test prompt"
        hash1 = _hash_prompt(prompt)
        hash2 = _hash_prompt(prompt)
        assert hash1 == hash2

    def test_hash_prompt_unique(self):
        """Test that different prompts produce different hashes."""
        hash1 = _hash_prompt("Prompt 1")
        hash2 = _hash_prompt("Prompt 2")
        assert hash1 != hash2

    def test_hash_prompt_length(self):
        """Test that hash is 64 characters (SHA-256 hex)."""
        hash_result = _hash_prompt("test")
        assert len(hash_result) == 64


class TestLireCache:
    """Tests for reading from cache."""

    def test_lire_cache_missing_entry(self):
        """Test that lire_cache returns None for non-existent prompt."""
        with patch('src.cache.GENAI_CACHE_FILE', Path(tempfile.gettempdir()) / "nonexistent_cache.json"):
            result = lire_cache("Unknown prompt")
            assert result is None

    def test_lire_cache_write_then_read(self):
        """Test writing and then reading a cache entry."""
        # Use a temporary file for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            with patch('src.cache.GENAI_CACHE_FILE', cache_file):
                prompt = "Test prompt"
                response = "Test response"

                # Write
                ecrire_cache(prompt, response)

                # Read
                result = lire_cache(prompt)
                assert result == response


class TestEcrireCache:
    """Tests for writing to cache."""

    def test_ecrire_cache_creates_file(self):
        """Test that ecrire_cache creates the cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            with patch('src.cache.GENAI_CACHE_FILE', cache_file):
                ecrire_cache("Test prompt", "Test response")
                assert cache_file.exists()

    def test_ecrire_cache_json_format(self):
        """Test that cache is saved in valid JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            with patch('src.cache.GENAI_CACHE_FILE', cache_file):
                ecrire_cache("Test prompt", "Test response")

                # Verify JSON is valid
                content = json.loads(cache_file.read_text(encoding="utf-8"))
                assert isinstance(content, dict)


class TestAppelAvecCache:
    """Tests for cached API calls (critical test for preventing duplicate API calls)."""

    def test_appel_avec_cache_first_call_invokes_api(self):
        """Test that first call invokes the API function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            with patch('src.cache.GENAI_CACHE_FILE', cache_file):
                mock_api = Mock(return_value="API Response")
                prompt = "Test prompt"

                result = appel_avec_cache(prompt, mock_api)

                # Should call the API
                mock_api.assert_called_once_with(prompt)
                assert result == "API Response"

    def test_appel_avec_cache_second_call_uses_cache(self):
        """Test that second call with same prompt DOES NOT invoke API (cache hit)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            with patch('src.cache.GENAI_CACHE_FILE', cache_file):
                mock_api = Mock(return_value="API Response")
                prompt = "Test prompt"

                # First call
                result1 = appel_avec_cache(prompt, mock_api)

                # Reset the mock to verify it's not called again
                mock_api.reset_mock()

                # Second call with same prompt
                result2 = appel_avec_cache(prompt, mock_api)

                # API should NOT be called on second invocation
                mock_api.assert_not_called()
                assert result2 == "API Response"
                assert result1 == result2

    def test_appel_avec_cache_different_prompts_invoke_api(self):
        """Test that different prompts cause new API calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            with patch('src.cache.GENAI_CACHE_FILE', cache_file):
                mock_api = Mock(side_effect=["Response 1", "Response 2"])

                # Call with different prompts
                result1 = appel_avec_cache("Prompt 1", mock_api)
                result2 = appel_avec_cache("Prompt 2", mock_api)

                # API should be called twice (once for each unique prompt)
                assert mock_api.call_count == 2
                assert result1 == "Response 1"
                assert result2 == "Response 2"

    def test_appel_avec_cache_caching_reduces_calls(self):
        """Integration test: verify that caching actually reduces API calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            with patch('src.cache.GENAI_CACHE_FILE', cache_file):
                mock_api = Mock(return_value="Cached Response")
                prompt = "Important prompt"

                # Make 5 identical calls
                for _ in range(5):
                    appel_avec_cache(prompt, mock_api)

                # API should be called only once (first call)
                assert mock_api.call_count == 1


class TestVideCache:
    """Tests for clearing cache."""

    def test_vider_cache_empties(self):
        """Test that vider_cache removes all entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            with patch('src.cache.GENAI_CACHE_FILE', cache_file):
                # Write some entries
                ecrire_cache("Prompt 1", "Response 1")
                ecrire_cache("Prompt 2", "Response 2")

                # Verify they're there
                assert taille_cache() == 2

                # Clear
                vider_cache()

                # Verify empty
                assert taille_cache() == 0

    def test_vider_cache_returns_count(self):
        """Test that vider_cache returns count of removed entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            with patch('src.cache.GENAI_CACHE_FILE', cache_file):
                ecrire_cache("Prompt 1", "Response 1")
                ecrire_cache("Prompt 2", "Response 2")

                count = vider_cache()
                assert count == 2


class TestTailleCache:
    """Tests for cache size."""

    def test_taille_cache_empty(self):
        """Test that taille_cache returns 0 for empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            with patch('src.cache.GENAI_CACHE_FILE', cache_file):
                size = taille_cache()
                assert size == 0

    def test_taille_cache_increments(self):
        """Test that taille_cache correctly counts entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            with patch('src.cache.GENAI_CACHE_FILE', cache_file):
                assert taille_cache() == 0

                ecrire_cache("Prompt 1", "Response 1")
                assert taille_cache() == 1

                ecrire_cache("Prompt 2", "Response 2")
                assert taille_cache() == 2

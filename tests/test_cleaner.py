"""Tests for dataset_curator.cleaner."""

import pytest
from dataset_curator.cleaner import DataCleaner


@pytest.fixture
def cleaner():
    return DataCleaner()


class TestClean:
    def test_strips_whitespace(self, cleaner):
        assert cleaner.clean("  hello  ") == "hello"

    def test_collapses_multiple_newlines(self, cleaner):
        assert cleaner.clean("a\n\n\n\nb") == "a\n\nb"

    def test_normalizes_unicode(self, cleaner):
        # NFKC normalization: fullwidth 'A' becomes regular 'A'
        result = cleaner.clean("\uff21\uff22\uff23")
        assert result == "ABC"

    def test_empty_string(self, cleaner):
        assert cleaner.clean("") == ""

    def test_already_clean(self, cleaner):
        assert cleaner.clean("already clean") == "already clean"


class TestIsValid:
    def test_valid_text(self, cleaner):
        assert cleaner.is_valid("This is a valid sentence.") is True

    def test_too_short(self, cleaner):
        assert cleaner.is_valid("short", min_length=10) is False

    def test_too_long(self, cleaner):
        assert cleaner.is_valid("a" * 10001, max_length=10000) is False

    def test_empty_string_invalid(self, cleaner):
        assert cleaner.is_valid("") is False

    def test_custom_min_length(self, cleaner):
        assert cleaner.is_valid("hey", min_length=3) is True
        assert cleaner.is_valid("he", min_length=3) is False

    def test_exactly_at_boundaries(self, cleaner):
        assert cleaner.is_valid("a" * 10, min_length=10) is True
        assert cleaner.is_valid("a" * 10000, max_length=10000) is True


class TestCleanBatch:
    def test_cleans_and_filters(self, cleaner):
        texts = ["  hello world  ", "ab", "This is a valid text here."]
        result = cleaner.clean_batch(texts)
        assert "hello world" in result
        assert "This is a valid text here." in result
        # "ab" is too short (< 10) after cleaning
        assert len(result) == 2

    def test_empty_list(self, cleaner):
        assert cleaner.clean_batch([]) == []

    def test_all_invalid(self, cleaner):
        assert cleaner.clean_batch(["a", "b", "c"]) == []

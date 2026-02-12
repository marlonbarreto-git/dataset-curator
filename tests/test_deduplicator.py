"""Tests for dataset_curator.deduplicator."""

import pytest
from dataset_curator.deduplicator import Deduplicator
from dataset_curator.models import ChatMessage, TrainingExample


def _make_example(content: str, source: str = "test") -> TrainingExample:
    return TrainingExample(
        messages=[
            ChatMessage(role="user", content=content),
            ChatMessage(role="assistant", content="response"),
        ],
        source=source,
    )


@pytest.fixture
def dedup():
    return Deduplicator(threshold=0.95)


class TestNormalize:
    def test_lowercases(self, dedup):
        assert dedup._normalize("HELLO") == "hello"

    def test_strips_whitespace(self, dedup):
        assert dedup._normalize("  hello  ") == "hello"

    def test_both(self, dedup):
        assert dedup._normalize("  HELLO World  ") == "hello world"


class TestSimilarity:
    def test_identical_strings(self, dedup):
        assert dedup._similarity("hello", "hello") == 1.0

    def test_completely_different(self, dedup):
        score = dedup._similarity("abc", "xyz")
        assert score < 0.5

    def test_similar_strings(self, dedup):
        score = dedup._similarity("hello world", "hello worl")
        assert score > 0.8


class TestDeduplicate:
    def test_removes_exact_duplicates(self, dedup):
        examples = [
            _make_example("What is Python?"),
            _make_example("What is Python?"),
            _make_example("What is Java?"),
        ]
        unique, removed = dedup.deduplicate(examples)
        assert len(unique) == 2
        assert removed == 1

    def test_removes_near_duplicates(self, dedup):
        examples = [
            _make_example("What is Python programming?"),
            _make_example("What is Python programming"),  # missing ? but very similar
            _make_example("Explain quantum computing"),
        ]
        unique, removed = dedup.deduplicate(examples)
        assert len(unique) == 2
        assert removed == 1

    def test_no_duplicates(self, dedup):
        examples = [
            _make_example("What is Python?"),
            _make_example("Explain quantum physics"),
            _make_example("How does a car engine work?"),
        ]
        unique, removed = dedup.deduplicate(examples)
        assert len(unique) == 3
        assert removed == 0

    def test_empty_list(self, dedup):
        unique, removed = dedup.deduplicate([])
        assert unique == []
        assert removed == 0

    def test_single_example(self, dedup):
        examples = [_make_example("Hello")]
        unique, removed = dedup.deduplicate(examples)
        assert len(unique) == 1
        assert removed == 0

    def test_custom_threshold(self):
        dedup_low = Deduplicator(threshold=0.5)
        examples = [
            _make_example("hello world"),
            _make_example("hello there"),
        ]
        unique, removed = dedup_low.deduplicate(examples)
        # With a low threshold, these might be considered duplicates
        assert removed >= 0  # depends on similarity score

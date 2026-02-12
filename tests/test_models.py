"""Tests for dataset_curator.models."""

import pytest
from dataset_curator.models import ChatMessage, DatasetStats, RawExample, TrainingExample


class TestRawExample:
    def test_create_with_required_fields(self):
        raw = RawExample(source="wiki", content="Some text")
        assert raw.source == "wiki"
        assert raw.content == "Some text"
        assert raw.metadata == {}

    def test_create_with_metadata(self):
        raw = RawExample(source="api", content="Data", metadata={"lang": "en"})
        assert raw.metadata == {"lang": "en"}

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            RawExample(source="wiki")


class TestChatMessage:
    def test_valid_roles(self):
        for role in ("system", "user", "assistant"):
            msg = ChatMessage(role=role, content="hello")
            assert msg.role == role

    def test_invalid_role_raises(self):
        with pytest.raises(Exception):
            ChatMessage(role="admin", content="hello")

    def test_content_stored(self):
        msg = ChatMessage(role="user", content="What is AI?")
        assert msg.content == "What is AI?"


class TestTrainingExample:
    def test_create_with_defaults(self):
        msgs = [ChatMessage(role="user", content="hi")]
        ex = TrainingExample(messages=msgs)
        assert ex.quality_score == 0.0
        assert ex.source == ""

    def test_create_with_all_fields(self):
        msgs = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="Hello!"),
        ]
        ex = TrainingExample(messages=msgs, quality_score=0.9, source="wiki")
        assert len(ex.messages) == 3
        assert ex.quality_score == 0.9
        assert ex.source == "wiki"


class TestDatasetStats:
    def test_create_stats(self):
        stats = DatasetStats(
            total_examples=100,
            avg_quality=0.85,
            sources={"wiki": 60, "api": 40},
        )
        assert stats.total_examples == 100
        assert stats.avg_quality == 0.85
        assert stats.sources == {"wiki": 60, "api": 40}
        assert stats.duplicates_removed == 0

    def test_create_stats_with_duplicates(self):
        stats = DatasetStats(
            total_examples=90,
            avg_quality=0.8,
            sources={"wiki": 90},
            duplicates_removed=10,
        )
        assert stats.duplicates_removed == 10

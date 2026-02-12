"""Tests for dataset_curator.pipeline."""

import pytest
from dataset_curator.cleaner import DataCleaner
from dataset_curator.converter import FormatConverter
from dataset_curator.deduplicator import Deduplicator
from dataset_curator.models import DatasetStats, RawExample, TrainingExample
from dataset_curator.pipeline import CurationPipeline


@pytest.fixture
def pipeline():
    return CurationPipeline(
        cleaner=DataCleaner(),
        converter=FormatConverter(),
        deduplicator=Deduplicator(threshold=0.95),
    )


class TestPipelineRun:
    def test_full_pipeline(self, pipeline):
        raws = [
            RawExample(source="wiki", content="What is machine learning and how does it work?"),
            RawExample(source="wiki", content="Explain deep learning in simple terms please."),
            RawExample(source="api", content="What is machine learning and how does it work?"),  # dup
        ]
        examples, stats = pipeline.run(raws)
        assert isinstance(stats, DatasetStats)
        assert isinstance(examples, list)
        assert all(isinstance(e, TrainingExample) for e in examples)
        assert stats.duplicates_removed >= 1
        assert stats.total_examples == len(examples)

    def test_filters_short_content(self, pipeline):
        raws = [
            RawExample(source="wiki", content="hi"),  # too short
            RawExample(source="wiki", content="This is a sufficiently long piece of content."),
        ]
        examples, stats = pipeline.run(raws)
        assert stats.total_examples == 1

    def test_empty_input(self, pipeline):
        examples, stats = pipeline.run([])
        assert examples == []
        assert stats.total_examples == 0
        assert stats.avg_quality == 0.0
        assert stats.duplicates_removed == 0

    def test_with_system_prompt(self, pipeline):
        raws = [
            RawExample(source="test", content="Explain the theory of relativity in detail."),
        ]
        examples, stats = pipeline.run(raws, system_prompt="You are a physics tutor.")
        assert any(
            m.role == "system" and m.content == "You are a physics tutor."
            for m in examples[0].messages
        )

    def test_stats_sources(self, pipeline):
        raws = [
            RawExample(source="wiki", content="First question about programming languages?"),
            RawExample(source="wiki", content="Second question about data structures?"),
            RawExample(source="api", content="Third question about algorithms and complexity?"),
        ]
        examples, stats = pipeline.run(raws)
        assert "wiki" in stats.sources
        assert "api" in stats.sources

    def test_stats_avg_quality(self, pipeline):
        raws = [
            RawExample(source="test", content="A valid example with enough content here."),
        ]
        examples, stats = pipeline.run(raws)
        assert 0.0 <= stats.avg_quality <= 1.0

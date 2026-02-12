"""Tests for dataset_curator.converter."""

import json

import pytest
from dataset_curator.converter import FormatConverter
from dataset_curator.models import ChatMessage, RawExample, TrainingExample


@pytest.fixture
def converter():
    return FormatConverter()


class TestToChatFormat:
    def test_basic_conversion(self, converter):
        raw = RawExample(source="wiki", content="What is Python?")
        result = converter.to_chat_format(raw)
        assert isinstance(result, TrainingExample)
        assert any(m.role == "user" and m.content == "What is Python?" for m in result.messages)
        assert any(m.role == "assistant" for m in result.messages)
        assert result.source == "wiki"

    def test_with_system_prompt(self, converter):
        raw = RawExample(source="api", content="Explain AI")
        result = converter.to_chat_format(raw, system_prompt="You are helpful.")
        roles = [m.role for m in result.messages]
        assert "system" in roles
        system_msg = next(m for m in result.messages if m.role == "system")
        assert system_msg.content == "You are helpful."

    def test_no_system_prompt_by_default(self, converter):
        raw = RawExample(source="test", content="Hello there")
        result = converter.to_chat_format(raw)
        roles = [m.role for m in result.messages]
        assert "system" not in roles


class TestToChatBatch:
    def test_converts_multiple(self, converter):
        raws = [
            RawExample(source="a", content="First question"),
            RawExample(source="b", content="Second question"),
        ]
        results = converter.to_chat_batch(raws)
        assert len(results) == 2
        assert all(isinstance(r, TrainingExample) for r in results)

    def test_empty_batch(self, converter):
        assert converter.to_chat_batch([]) == []

    def test_batch_with_system_prompt(self, converter):
        raws = [RawExample(source="s", content="Q")]
        results = converter.to_chat_batch(raws, system_prompt="Be concise.")
        assert any(m.role == "system" for m in results[0].messages)


class TestToJsonl:
    def test_produces_valid_jsonl(self, converter):
        msgs = [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
        examples = [TrainingExample(messages=msgs, source="test")]
        result = converter.to_jsonl(examples)
        lines = result.strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "messages" in parsed

    def test_multiple_examples(self, converter):
        msgs = [ChatMessage(role="user", content="q")]
        examples = [
            TrainingExample(messages=msgs, source="a"),
            TrainingExample(messages=msgs, source="b"),
        ]
        result = converter.to_jsonl(examples)
        lines = result.strip().split("\n")
        assert len(lines) == 2

    def test_empty_list(self, converter):
        assert converter.to_jsonl([]) == ""

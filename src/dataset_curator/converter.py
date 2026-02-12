"""Format conversion utilities for dataset curation."""

import json

from dataset_curator.models import ChatMessage, RawExample, TrainingExample


class FormatConverter:
    def to_chat_format(self, raw: RawExample, system_prompt: str = "") -> TrainingExample:
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=raw.content))
        messages.append(ChatMessage(role="assistant", content=""))
        return TrainingExample(messages=messages, source=raw.source)

    def to_chat_batch(
        self, raws: list[RawExample], system_prompt: str = ""
    ) -> list[TrainingExample]:
        return [self.to_chat_format(raw, system_prompt) for raw in raws]

    def to_jsonl(self, examples: list[TrainingExample]) -> str:
        if not examples:
            return ""
        lines = [json.dumps(ex.model_dump(), ensure_ascii=False) for ex in examples]
        return "\n".join(lines)

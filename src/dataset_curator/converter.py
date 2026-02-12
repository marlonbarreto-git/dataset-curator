"""Format conversion utilities for dataset curation."""

import json

from dataset_curator.models import ChatMessage, RawExample, TrainingExample


class FormatConverter:
    """Converts raw examples into chat-formatted training data."""

    def to_chat_format(self, raw: RawExample, system_prompt: str = "") -> TrainingExample:
        """Convert a single raw example into a chat-style training example.

        Args:
            raw: The raw example to convert.
            system_prompt: Optional system message prepended to the conversation.

        Returns:
            A formatted training example with chat messages.
        """
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=raw.content))
        messages.append(ChatMessage(role="assistant", content=""))
        return TrainingExample(messages=messages, source=raw.source)

    def to_chat_batch(
        self, raws: list[RawExample], system_prompt: str = ""
    ) -> list[TrainingExample]:
        """Convert a batch of raw examples into chat-formatted training examples.

        Args:
            raws: List of raw examples.
            system_prompt: Optional system message prepended to each conversation.

        Returns:
            List of formatted training examples.
        """
        return [self.to_chat_format(raw, system_prompt) for raw in raws]

    def to_jsonl(self, examples: list[TrainingExample]) -> str:
        """Serialize training examples to JSONL format.

        Args:
            examples: List of training examples.

        Returns:
            A newline-delimited JSON string, or empty string if no examples.
        """
        if not examples:
            return ""
        lines = [json.dumps(ex.model_dump(), ensure_ascii=False) for ex in examples]
        return "\n".join(lines)

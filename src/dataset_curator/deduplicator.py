"""Deduplication utilities for dataset curation."""

from difflib import SequenceMatcher

from dataset_curator.models import TrainingExample


class Deduplicator:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def _normalize(self, text: str) -> str:
        return text.strip().lower()

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _extract_text(self, example: TrainingExample) -> str:
        user_msgs = [m.content for m in example.messages if m.role == "user"]
        return " ".join(user_msgs)

    def deduplicate(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], int]:
        if not examples:
            return [], 0

        unique: list[TrainingExample] = []
        seen_texts: list[str] = []

        for example in examples:
            text = self._normalize(self._extract_text(example))
            is_dup = False
            for seen in seen_texts:
                if self._similarity(text, seen) >= self.threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(example)
                seen_texts.append(text)

        removed = len(examples) - len(unique)
        return unique, removed

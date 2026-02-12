"""Deduplication utilities for dataset curation."""

from difflib import SequenceMatcher

from dataset_curator.models import TrainingExample

DEFAULT_SIMILARITY_THRESHOLD: float = 0.95


class Deduplicator:
    """Removes near-duplicate training examples using sequence similarity."""

    def __init__(self, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> None:
        """Initialize the deduplicator.

        Args:
            threshold: Similarity ratio at or above which two texts are
                considered duplicates.
        """
        self.threshold = threshold

    def _normalize(self, text: str) -> str:
        """Lowercase and strip whitespace from text."""
        return text.strip().lower()

    def _similarity(self, a: str, b: str) -> float:
        """Compute the similarity ratio between two strings."""
        return SequenceMatcher(None, a, b).ratio()

    def _extract_text(self, example: TrainingExample) -> str:
        """Extract concatenated user message content from a training example."""
        user_msgs = [m.content for m in example.messages if m.role == "user"]
        return " ".join(user_msgs)

    def deduplicate(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], int]:
        """Remove near-duplicate examples from a list.

        Args:
            examples: List of training examples to deduplicate.

        Returns:
            A tuple of (unique examples, number of duplicates removed).
        """
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

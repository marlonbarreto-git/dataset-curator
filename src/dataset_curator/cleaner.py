"""Text cleaning utilities for dataset curation."""

import re
import unicodedata

DEFAULT_MIN_LENGTH: int = 10
DEFAULT_MAX_LENGTH: int = 10000
MAX_CONSECUTIVE_NEWLINES: int = 3


class DataCleaner:
    """Normalizes and validates raw text for downstream processing."""

    def clean(self, text: str) -> str:
        """Apply unicode normalization and whitespace cleanup to text.

        Args:
            text: Raw input text.

        Returns:
            Cleaned and normalized text.
        """
        text = unicodedata.normalize("NFKC", text)
        text = text.strip()
        text = re.sub(rf"\n{{{MAX_CONSECUTIVE_NEWLINES},}}", "\n\n", text)
        return text

    def is_valid(
        self,
        text: str,
        min_length: int = DEFAULT_MIN_LENGTH,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> bool:
        """Check whether text length falls within acceptable bounds.

        Args:
            text: Text to validate.
            min_length: Minimum acceptable character count.
            max_length: Maximum acceptable character count.

        Returns:
            True if the text length is within bounds.
        """
        return min_length <= len(text) <= max_length

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean a batch of texts, discarding any that fail validation.

        Args:
            texts: List of raw text strings.

        Returns:
            List of cleaned, valid text strings.
        """
        results: list[str] = []
        for text in texts:
            cleaned = self.clean(text)
            if self.is_valid(cleaned):
                results.append(cleaned)
        return results

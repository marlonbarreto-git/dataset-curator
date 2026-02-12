"""Text cleaning utilities for dataset curation."""

import re
import unicodedata


class DataCleaner:
    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def is_valid(self, text: str, min_length: int = 10, max_length: int = 10000) -> bool:
        return min_length <= len(text) <= max_length

    def clean_batch(self, texts: list[str]) -> list[str]:
        results = []
        for text in texts:
            cleaned = self.clean(text)
            if self.is_valid(cleaned):
                results.append(cleaned)
        return results

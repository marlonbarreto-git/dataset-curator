"""Main curation pipeline orchestrating all processing steps."""

from collections import Counter

from dataset_curator.cleaner import DataCleaner
from dataset_curator.converter import FormatConverter
from dataset_curator.deduplicator import Deduplicator
from dataset_curator.models import DatasetStats, RawExample, TrainingExample


class CurationPipeline:
    def __init__(
        self,
        cleaner: DataCleaner,
        converter: FormatConverter,
        deduplicator: Deduplicator,
    ):
        self.cleaner = cleaner
        self.converter = converter
        self.deduplicator = deduplicator

    def run(
        self, raws: list[RawExample], system_prompt: str = ""
    ) -> tuple[list[TrainingExample], DatasetStats]:
        if not raws:
            return [], DatasetStats(
                total_examples=0, avg_quality=0.0, sources={}, duplicates_removed=0
            )

        # Clean: filter out invalid content
        cleaned_raws = [
            raw for raw in raws if self.cleaner.is_valid(self.cleaner.clean(raw.content))
        ]

        # Convert to chat format
        examples = self.converter.to_chat_batch(cleaned_raws, system_prompt)

        # Deduplicate
        unique_examples, duplicates_removed = self.deduplicator.deduplicate(examples)

        # Compute quality scores based on content length (simple heuristic)
        for ex in unique_examples:
            user_content = " ".join(m.content for m in ex.messages if m.role == "user")
            ex.quality_score = min(len(user_content) / 500.0, 1.0)

        # Compute stats
        source_counts = dict(Counter(ex.source for ex in unique_examples))
        avg_quality = (
            sum(ex.quality_score for ex in unique_examples) / len(unique_examples)
            if unique_examples
            else 0.0
        )

        stats = DatasetStats(
            total_examples=len(unique_examples),
            avg_quality=round(avg_quality, 4),
            sources=source_counts,
            duplicates_removed=duplicates_removed,
        )

        return unique_examples, stats

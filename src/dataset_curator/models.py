"""Pydantic models for the dataset curation pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class RawExample(BaseModel):
    """A raw, unprocessed example ingested from an external data source."""

    source: str = Field(description="Identifier of the data source this example came from")
    content: str = Field(description="Raw text content of the example")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata associated with the example",
    )


class ChatMessage(BaseModel):
    """A single message within a multi-turn chat conversation."""

    role: Literal["system", "user", "assistant"] = Field(
        description="Role of the message author"
    )
    content: str = Field(description="Text content of the message")


class TrainingExample(BaseModel):
    """A fully formatted training example ready for model fine-tuning."""

    messages: list[ChatMessage] = Field(
        description="Ordered list of chat messages forming the conversation"
    )
    quality_score: float = Field(
        default=0.0, description="Heuristic quality score between 0.0 and 1.0"
    )
    source: str = Field(default="", description="Identifier of the originating data source")


class DatasetStats(BaseModel):
    """Aggregate statistics for a curated dataset."""

    total_examples: int = Field(description="Number of examples after curation")
    avg_quality: float = Field(description="Mean quality score across all examples")
    sources: dict[str, int] = Field(
        description="Mapping of source identifiers to example counts"
    )
    duplicates_removed: int = Field(
        default=0, description="Number of duplicate examples removed during curation"
    )

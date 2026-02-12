"""Pydantic models for the dataset curation pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class RawExample(BaseModel):
    source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class TrainingExample(BaseModel):
    messages: list[ChatMessage]
    quality_score: float = 0.0
    source: str = ""


class DatasetStats(BaseModel):
    total_examples: int
    avg_quality: float
    sources: dict[str, int]
    duplicates_removed: int = 0

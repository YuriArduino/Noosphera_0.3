"""Logical model for semantic experience — validation only."""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pydantic import Field, ConfigDict
from sqlmodel import SQLModel


class SemanticExperienceModel(SQLModel):
    """
    Logical representation of a vectorized experience.
    Used for embedding generation and semantic retrieval logic.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    session_id: str = Field(..., max_length=64)
    content: str = Field(..., min_length=1, max_length=8192)
    title: Optional[str] = Field(default=None, max_length=256)
    category: Optional[str] = Field(default=None, max_length=64)

    tags: List[str] = Field(default_factory=list)
    metadata_json: Dict[str, Any] = Field(default_factory=dict)

    # Embedding as list for logical layer (schema converts to Vector)
    embedding: Optional[List[float]] = Field(default=None)

    # Metrics
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    usage_count: int = Field(default=0)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

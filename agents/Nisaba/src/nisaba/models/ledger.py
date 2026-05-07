"""Logical models for audit and interaction tracking — validation only."""

from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import Field, ConfigDict
from sqlmodel import SQLModel


class DecisionLedgerModel(SQLModel):
    """Logical representation of a strategic decision."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    session_id: str = Field(..., max_length=64)
    trace_id: Optional[str] = Field(default=None, max_length=64)
    action: str = Field(..., max_length=32)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    strategy: Optional[str] = Field(default=None, max_length=64)
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    hitl_triggered: bool = Field(default=False)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InteractionLedgerModel(SQLModel):
    """
    Logical representation of a user-agent interaction.
    Used for quality tracking, feedback loops, and learning.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    session_id: str = Field(..., max_length=64)

    # Content
    user_message: str = Field(..., min_length=1, max_length=8192)
    agent_response: str = Field(..., min_length=1, max_length=8192)

    # Metrics
    tokens_used: Optional[int] = Field(default=None, ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0)
    success: bool = Field(default=True)
    feedback_score: Optional[float] = Field(default=None, ge=0.0, le=10.0)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

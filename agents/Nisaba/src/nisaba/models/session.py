"""Logical model for session state — validation only, no DB persistence."""

from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import Field, ConfigDict
from sqlmodel import SQLModel


class SessionStateModel(SQLModel):
    """
    Logical representation of a conversation session.
    Used for validation, serialization, and business logic.

    Note: This model is frozen and has no table=True.
    For persistence, use schema/tables.py → SessionStateTable.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    session_id: str = Field(..., max_length=64, description="Unique session identifier")
    user_id: Optional[str] = Field(default=None, max_length=64)

    state_data: Optional[Dict[str, Any]] = Field(default_factory=dict)

    ttl_seconds: Optional[int] = Field(default=3600, ge=60, le=86400)

    # Audit (read-only in logical layer)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

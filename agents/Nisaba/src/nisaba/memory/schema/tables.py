"""
Nisaba Schema — Single Source of Truth (SSOT) for Database Models.

All SQLModel table definitions live here. Alembic reads this file exclusively
for auto-generating migrations. This ensures DRY and prevents schema drift.

Pattern: Mirrors Glyphar's schema/tables.py for architectural consistency.
"""

from typing import Optional, List
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, JSON

# =============================================================================
# BASE CONFIGURATION
# =============================================================================
# Shared schema for all Nisaba tables (logical isolation in shared DB)
TABLE_SCHEMA = "public"
TABLE_ARGS = {"schema": TABLE_SCHEMA}

# =============================================================================
# SESSION STATE — Short-Term Memory (PostgreSQL JSONB)
# =============================================================================


class SessionState(SQLModel, table=True):
    """
    Stores conversation state per session with flexible JSONB payload.
    Used by LangGraph checkpointer and session restoration.
    """

    __tablename__ = "session_state"
    __table_args__ = TABLE_ARGS

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, unique=True, max_length=64)
    user_id: Optional[str] = Field(default=None, index=True, max_length=64)

    # JSONB for flexible, evolving state structure
    state_data: dict = Field(sa_column=Column(JSON), default_factory=dict)

    # Lifecycle
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: Optional[int] = Field(default=3600, description="Auto-expiry in seconds")


# =============================================================================
# SEMANTIC EXPERIENCE — Long-Term Memory (pgvector)
# =============================================================================


class SemanticExperience(SQLModel, table=True):
    """
    Vectorized experiences for semantic similarity search.
    Enables Nisaba to 'recall' similar past situations for informed responses.
    """

    __tablename__ = "semantic_experience"
    __table_args__ = TABLE_ARGS

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, max_length=64)

    # Content for embedding
    content: str = Field(max_length=8192)
    title: Optional[str] = Field(default=None, max_length=256)
    category: Optional[str] = Field(default=None, index=True, max_length=64)

    # Metadata
    tags: List[str] = Field(sa_column=Column(JSON), default_factory=list)
    metadata_json: dict = Field(sa_column=Column(JSON), default_factory=dict)

    # pgvector embedding (dimension from llm_settings)
    embedding: Optional[List[float]] = Field(sa_column=Column(Vector(768)), default=None)

    # Usage tracking for relevance ranking
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    usage_count: int = Field(default=0)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# DECISION LEDGER — Audit Trail (Optional, for learning)
# =============================================================================


class DecisionLedger(SQLModel, table=True):
    """
    Logs strategic decisions for audit, debugging, and offline learning.
    Not required for core operation but valuable for agent evolution.
    """

    __tablename__ = "decision_ledger"
    __table_args__ = TABLE_ARGS

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, max_length=64)
    trace_id: Optional[str] = Field(default=None, index=True, max_length=64)

    # Decision context
    action: str = Field(max_length=32)  # e.g., "respond", "retrieve", "escalate"
    confidence: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    strategy: Optional[str] = Field(default=None, max_length=64)

    # Flexible metadata
    metadata_json: dict = Field(sa_column=Column(JSON), default_factory=dict)
    hitl_triggered: bool = Field(default=False)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InteractionLedgerTable(InteractionLedgerModel, TimestampMixin, table=True):
    """
    Physical table for interaction tracking.
    Enables feedback analysis and quality improvement loops.
    """

    __tablename__ = "interaction_ledger"
    __table_args__ = TABLE_ARGS
    model_config = ConfigDict(frozen=False, extra="ignore")

    id: Optional[int] = Field(default=None, primary_key=True)


# =============================================================================
# EXPORTS FOR ALEMBIC
# =============================================================================
# Alembic's env.py imports this to get target_metadata
Base = SQLModel
__all__ = ["SessionState", "SemanticExperience", "DecisionLedger", "Base"]

"""
Single Source of Truth (SSOT) for Nisaba Database Schema.

Materializes logical domain models into physical PostgreSQL tables.
Exclusive entry point for SQLAlchemy sessions, Alembic migrations,
and vectorized memory operations.

Pattern: Mirrors glyphar/schema/tables.py for architectural consistency.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import ConfigDict
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

# Import logical domain models (frozen, validation-only)
from nisaba.models.session import SessionStateModel
from nisaba.models.ledger import DecisionLedgerModel, InteractionLedgerModel
from nisaba.models.memory import SemanticExperienceModel

# =============================================================================
# SCHEMA CONFIGURATION
# =============================================================================
TABLE_SCHEMA = "nisaba"
TABLE_ARGS = {"schema": TABLE_SCHEMA}

# Registry for SQLAlchemy & Alembic auto-discovery
metadata = SQLModel.metadata


# =============================================================================
# MIXINS
# =============================================================================


class TimestampMixin(SQLModel):
    """Automatic audit timestamps with DB-level defaults."""

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={
            "server_default": text("CURRENT_TIMESTAMP"),
            "onupdate": text("CURRENT_TIMESTAMP"),
        },
    )


# =============================================================================
# PHYSICAL TABLES (Materialized from Logical Models)
# =============================================================================


class SessionStateTable(SessionStateModel, TimestampMixin, table=True):
    """
    Physical table for conversation state persistence.
    Uses JSONB for flexible, evolving session payloads.
    """

    __tablename__ = "session_state"
    __table_args__ = TABLE_ARGS
    model_config = ConfigDict(frozen=False, extra="ignore")

    id: Optional[int] = Field(default=None, primary_key=True)

    # ✅ CORREÇÃO: dois-pontos após o nome do campo
    state_data: Optional[Dict[str, Any]] = Field(default_factory=dict, sa_column=Column(JSONB))


class DecisionLedgerTable(DecisionLedgerModel, TimestampMixin, table=True):
    """
    Physical table for strategic decision audit trail.
    Enables offline learning and HITL compliance tracking.
    """

    __tablename__ = "decision_ledger"
    __table_args__ = TABLE_ARGS
    model_config = ConfigDict(frozen=False, extra="ignore")

    id: Optional[int] = Field(default=None, primary_key=True)

    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))


class InteractionLedgerTable(InteractionLedgerModel, TimestampMixin, table=True):
    """
    Physical table for user-agent interaction tracking.
    Stores quality metrics for feedback loops and latency analysis.
    """

    __tablename__ = "interaction_ledger"
    __table_args__ = TABLE_ARGS
    model_config = ConfigDict(frozen=False, extra="ignore")

    id: Optional[int] = Field(default=None, primary_key=True)


class SemanticExperienceTable(SemanticExperienceModel, TimestampMixin, table=True):
    """
    Physical table for semantic memory with pgvector support.
    Enables similarity search via embedding vector for contextual recall.
    """

    __tablename__ = "semantic_experience"
    __table_args__ = TABLE_ARGS
    model_config = ConfigDict(frozen=False, extra="ignore")

    id: Optional[int] = Field(default=None, primary_key=True)

    # ✅ CORREÇÃO: dois-pontos após o nome do campo
    tags: List[str] = Field(default_factory=list, sa_column=Column(JSONB))
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))

    # pgvector embedding (768D aligned with llm_settings.EMBEDDING_DIMENSION)
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(768)))


# =============================================================================
# EXPORTS FOR ALEMBIC & INTERNAL USE
# =============================================================================

# Registry for SQLAlchemy/Alembic auto-discovery
metadata = SQLModel.metadata

# Alias for Alembic compatibility (mirrors Glyphar pattern)
Base = SQLModel

__all__ = [
    "metadata",
    "Base",
    "SessionStateTable",
    "DecisionLedgerTable",
    "InteractionLedgerTable",
    "SemanticExperienceTable",
]

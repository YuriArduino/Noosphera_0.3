"""
Cognitive Ledger for the Thoth Agent.

Responsible for:
- Auditing agent decisions (Decision Ledger)
- Tracking LLM-based text corrections (Correction Ledger)
- Storing outcomes for causal analysis and learning (Semantic Experience)
- Integrating with pgvector for similarity search

This version preserves 100% of the original logic while upgrading to SQLModel.
"""

from typing import Optional, List, cast
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, Session, select, create_engine
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column

# Shared Domain Types
from glyphar.core.types import LayoutType, PageQuality

# Agent Settings
from thoth.config import memory_settings

# ==========================================================
# LEDGER SCHEMAS (Tables)
# ==========================================================


class DecisionLedger(SQLModel, table=True):  # type: ignore[call-arg]
    """Logs every high-level decision made by Thoth."""

    __tablename__ = "thoth_decision_ledger"

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: str = Field(index=True)
    document_hash: str = Field(index=True)
    action: str
    strategy: Optional[str] = None
    avg_confidence: float
    attempts: int = 1
    execution_step: str
    hitl_triggered: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CorrectionLedger(SQLModel, table=True):  # type: ignore[call-arg]
    """Tracks LLM-based corrections and their perceived quality gain."""

    __tablename__ = "thoth_correction_ledger"

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: str = Field(index=True)
    document_hash: str = Field(index=True)
    model_name: str
    original_confidence: float
    final_confidence: float
    processing_time: float
    urgency: Optional[str] = None
    success: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SemanticExperience(SQLModel, table=True):  # type: ignore[call-arg]
    """
    The 'Learning' table.
    Stores results indexed by vectors for similarity search.
    """

    __tablename__ = "thoth_semantic_experience"

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: str = Field(index=True)
    document_hash: str = Field(index=True)
    layout_type: Optional[LayoutType] = None
    page_quality: Optional[PageQuality] = None
    error_type: Optional[str] = None
    strategy_used: str
    confidence: float
    snippet: Optional[str] = None

    # pgvector embedding: Allowing Thoth to query 'similar experiences'
    # Defaulting to 1536 dimensions (Standard for many LLMs)
    embedding: Optional[List[float]] = Field(sa_column=Column(Vector(1536)), default=None)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==========================================================
# LEDGER MANAGER
# ==========================================================


class ThothLedger:
    """Manages persistence for the Agent's cognitive history."""

    def __init__(self):
        self.enabled = memory_settings.LEDGER_ENABLED
        self.db_url = memory_settings.THOTH_DATABASE_URL  # Should point to thoth_db in .env

        if not self.enabled:
            self.engine = None
            return

        self.engine = create_engine(self.db_url)

        # Handle automatic table creation if enabled (Legacy behavior replacement)
        if memory_settings.LEDGER_AUTO_MIGRATE:
            SQLModel.metadata.create_all(self.engine)

    def log_decision(
        self,
        document_id: str,
        document_hash: str,
        action: str,
        strategy: Optional[str],
        avg_confidence: float,
        attempts: int,
        execution_step: str,
        hitl_triggered: bool,
    ):
        """Log a Thoth decision event to the ledger."""
        if not self.enabled:
            return

        with Session(self.engine) as session:
            entry = DecisionLedger(
                document_id=document_id,
                document_hash=document_hash,
                action=action,
                strategy=strategy,
                avg_confidence=avg_confidence,
                attempts=attempts,
                execution_step=execution_step,
                hitl_triggered=hitl_triggered,
            )
            session.add(entry)
            session.commit()

    def log_correction(
        self,
        document_id: str,
        document_hash: str,
        model_name: str,
        original_confidence: float,
        final_confidence: float,
        processing_time: float,
        urgency: Optional[str],
        success: bool,
    ):
        """Log an LLM correction event to the ledger."""
        if not self.enabled:
            return

        with Session(self.engine) as session:
            entry = CorrectionLedger(
                document_id=document_id,
                document_hash=document_hash,
                model_name=model_name,
                original_confidence=original_confidence,
                final_confidence=final_confidence,
                processing_time=processing_time,
                urgency=urgency,
                success=success,
            )
            session.add(entry)
            session.commit()

    def log_semantic_experience(
        self,
        document_id: str,
        document_hash: str,
        layout_type: Optional[LayoutType],
        page_quality: Optional[PageQuality],
        error_type: Optional[str],
        snippet: Optional[str],
        strategy_used: str,
        confidence: float,
        embedding: Optional[List[float]] = None,
    ):
        """Log a semantic experience for future learning and retrieval."""
        if not self.enabled:
            return

        with Session(self.engine) as session:
            entry = SemanticExperience(
                document_id=document_id,
                document_hash=document_hash,
                layout_type=layout_type,
                page_quality=page_quality,
                error_type=error_type,
                snippet=snippet,
                strategy_used=strategy_used,
                confidence=confidence,
                embedding=embedding,
            )
            session.add(entry)
            session.commit()

    def find_similar_experiences(self, query_embedding: List[float], limit: int = 3):
        """
        Uses pgvector to find similar historical cases.
        This enables Thoth to 'recall' what worked in similar situations.
        """
        if not self.enabled:
            return []

        with Session(self.engine) as session:
            statement = (
                select(SemanticExperience)
                .order_by(cast(Vector, SemanticExperience.embedding).l2_distance(query_embedding))
                .limit(limit)
            )
            return session.exec(statement).all()


def close(self):
    """Dispose of the connection pool (compatibilidade)."""
    if self.engine:
        self.engine.dispose()

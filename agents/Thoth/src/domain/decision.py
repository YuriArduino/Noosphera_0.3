"""
Decision Models — Thoth Agent Autonomy.

Models representing decisions made by the agent.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, computed_field

from .common import ThothAction, GlypharStrategy, PageQuality
from .ocr import OCROutput


class QualityMetrics(BaseModel):
    """Quality metrics that inform decisions."""

    avg_confidence: float = Field(..., ge=0.0, le=100.0)
    poor_pages_count: int = Field(..., ge=0)
    fair_pages_count: int = Field(..., ge=0)
    excellent_pages_count: int = Field(..., ge=0)
    min_confidence: float = Field(..., ge=0.0, le=100.0)
    max_confidence: float = Field(..., ge=0.0, le=100.0)

    model_config = {"frozen": True}


class DecisionContext(BaseModel):
    """
    Context for a decision made by Thoth.

    Captures the state that led to the decision.
    """

    doc_hash: str = Field(..., description="Document SHA256 hash")
    doc_name: str = Field(..., description="Document filename")
    current_strategy: GlypharStrategy = Field(...)
    attempt_number: int = Field(..., ge=0, description="Reprocessing attempt count")
    quality_metrics: QualityMetrics = Field(...)
    poor_page_numbers: Optional[List[int]] = Field(default=None)

    model_config = {"frozen": True}


class ThothDecision(BaseModel):
    """
    A decision made by Thoth agent.

    Immutable record of agent autonomy.
    """

    context: DecisionContext = Field(..., description="Decision context")
    action: ThothAction = Field(..., description="Action to take")
    reason: str = Field(..., description="Human-readable reason")
    next_strategy: Optional[GlypharStrategy] = Field(default=None)
    target_pages: Optional[List[int]] = Field(default=None)
    llm_input: Optional[str] = Field(default=None)
    decided_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": True}

    @computed_field
    @property
    def requires_reprocessing(self) -> bool:
        """Check if decision requires reprocessing."""
        return self.action == ThothAction.REPROCESS

    @computed_field
    @property
    def requires_llm_correction(self) -> bool:
        """Check if decision requires LLM correction."""
        return self.action == ThothAction.CORRECT

    @computed_field
    @property
    def is_final(self) -> bool:
        """Check if decision is final (approve or reject)."""
        return self.action in [ThothAction.APPROVE, ThothAction.REJECT]

    def to_dict(self) -> dict:
        """Convert to dictionary for state storage."""
        return {
            "doc_hash": self.context.doc_hash,
            "doc_name": self.context.doc_name,
            "action": self.action.value,
            "reason": self.reason,
            "metrics": {
                "avg_confidence": self.context.quality_metrics.avg_confidence,
                "poor_pages_count": self.context.quality_metrics.poor_pages_count,
                "fair_pages_count": self.context.quality_metrics.fair_pages_count,
            },
            "target_pages": self.target_pages,
            "current_strategy": self.context.current_strategy.value,
            "next_strategy": self.next_strategy.value if self.next_strategy else None,
            "llm_input": self.llm_input,
        }


class DecisionHistory(BaseModel):
    """
    Complete decision history for a document.

    Used for learning and audit trail.
    """

    doc_hash: str = Field(...)
    decisions: List[ThothDecision] = Field(default_factory=list)
    final_action: Optional[ThothAction] = Field(default=None)
    total_reprocess_attempts: int = Field(default=0)
    final_confidence: Optional[float] = Field(default=None)

    model_config = {"frozen": True}

    def add_decision(self, decision: ThothDecision) -> "DecisionHistory":
        """Add a decision to history (returns new immutable instance)."""
        new_decisions = self.decisions + [decision]
        return DecisionHistory(
            doc_hash=self.doc_hash,
            decisions=new_decisions,
            final_action=decision.action if decision.is_final else self.final_action,
            total_reprocess_attempts=self.total_reprocess_attempts
            + (1 if decision.requires_reprocessing else 0),
            final_confidence=decision.context.quality_metrics.avg_confidence,
        )

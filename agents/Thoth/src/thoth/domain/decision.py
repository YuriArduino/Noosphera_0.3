"""
Decision Models — Thoth Agent Autonomy.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, computed_field

from .common import (
    ThothAction,
    GlypharStrategy,
    HashSHA256,
)
from .ocr import OCROutput


# ================================================================
# QUALITY METRICS
# ================================================================
class QualityMetrics(BaseModel):
    """
    Immutable snapshot of quality statistics used for decision-making.
    """

    avg_confidence: float = Field(..., ge=0.0, le=100.0)
    poor_pages_count: int = Field(..., ge=0)
    fair_pages_count: int = Field(..., ge=0)
    excellent_pages_count: int = Field(..., ge=0)
    min_confidence: float = Field(..., ge=0.0, le=100.0)
    max_confidence: float = Field(..., ge=0.0, le=100.0)

    model_config = {"frozen": True}


# ================================================================
# DECISION CONTEXT
# ================================================================
class DecisionContext(BaseModel):
    """
    Immutable context snapshot for a Thoth decision.
    """

    ocr_output: OCROutput
    quality_metrics: QualityMetrics
    current_strategy: GlypharStrategy
    attempt_number: int = Field(..., ge=0)

    model_config = {"frozen": True}

    @computed_field
    @property
    def doc_hash(self) -> HashSHA256:
        """Hash of the document, used for tracking and idempotency."""
        return self.ocr_output.file_metadata.hash_sha256

    @computed_field
    @property
    def doc_name(self) -> str:
        """Original document name, for logging and user-friendly output."""
        return self.ocr_output.file_metadata.name


# ================================================================
# THOTH DECISION (DOMAIN EVENT)
# ================================================================
class ThothDecision(BaseModel):
    """
    Immutable domain event representing a Thoth decision.
    """

    context: DecisionContext
    action: ThothAction
    reason: str

    next_strategy: Optional[GlypharStrategy] = None
    target_pages: Optional[List[int]] = None
    llm_input: Optional[str] = None

    decided_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": True}

    @computed_field
    @property
    def requires_reprocessing(self) -> bool:
        """Indicates if this decision requires another OCR attempt."""
        return self.action == ThothAction.REPROCESS

    @computed_field
    @property
    def requires_llm_correction(self) -> bool:
        """Indicates if this decision requires an LLM correction."""
        return self.action == ThothAction.CORRECT

    @computed_field
    @property
    def is_final(self) -> bool:
        """Indicates if this decision is a final action with no further steps."""
        return self.action.is_terminal

    def to_state_dict(self) -> dict:
        """
        Projection for workflow/state persistence.
        """
        return {
            "doc_hash": self.context.doc_hash,
            "doc_name": self.context.doc_name,
            "action": self.action.value,
            "reason": self.reason,
            "metrics": self.context.quality_metrics.model_dump(),
            "target_pages": self.target_pages,
            "current_strategy": self.context.current_strategy.value,
            "next_strategy": self.next_strategy.value if self.next_strategy else None,
            "llm_input": self.llm_input,
        }


# ================================================================
# DECISION HISTORY (AGGREGATE)
# ================================================================
class DecisionHistory(BaseModel):
    """
    Aggregate of all decisions made for a document.
    """

    doc_hash: HashSHA256
    decisions: List[ThothDecision] = Field(default_factory=list)

    final_action: Optional[ThothAction] = None
    total_reprocess_attempts: int = 0
    final_confidence: Optional[float] = None

    model_config = {"frozen": True}

    def add_decision(self, decision: ThothDecision) -> "DecisionHistory":
        """
        Return new immutable history with appended decision.
        """
        return DecisionHistory(
            doc_hash=self.doc_hash,
            decisions=self.decisions + [decision],
            final_action=decision.action if decision.is_final else self.final_action,
            total_reprocess_attempts=self.total_reprocess_attempts
            + (1 if decision.requires_reprocessing else 0),
            final_confidence=decision.context.quality_metrics.avg_confidence,
        )

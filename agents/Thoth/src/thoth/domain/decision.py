"""
Decision Models — Thoth Agent Autonomy and Reasoning.

This module defines the immutable domain entities and events used by the
Agent to evaluate OCR results and determine the next course of action.
"""

from datetime import datetime, timezone
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field, computed_field, ConfigDict

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
    An analytical snapshot of quality statistics used for strategic evaluation.

    This object flattens complex OCR statistics into a format optimized
    for the Agent's decision logic and Neo4j causal mapping.
    """

    avg_confidence: float = Field(..., ge=0.0, le=100.0)
    poor_pages_count: int = Field(..., ge=0)
    fair_pages_count: int = Field(..., ge=0)
    excellent_pages_count: int = Field(..., ge=0)
    min_confidence: float = Field(..., ge=0.0, le=100.0)
    max_confidence: float = Field(..., ge=0.0, le=100.0)

    model_config = ConfigDict(frozen=True)


# ================================================================
# DECISION CONTEXT
# ================================================================
class DecisionContext(BaseModel):
    """
    The situational context surrounding a specific decision.

    Binds the raw perceptual data (OCR Output) with the operational
    state (Strategy and Attempt Number).
    """

    ocr_output: OCROutput
    quality_metrics: QualityMetrics
    current_strategy: GlypharStrategy
    attempt_number: int = Field(..., ge=0)

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def doc_hash(self) -> HashSHA256:
        """SST reference for tracking and idempotency across Noosphera tools."""
        return self.ocr_output.file_metadata.hash_sha256

    @computed_field
    @property
    def doc_name(self) -> str:
        """The original filename for logging and trace visibility."""
        return self.ocr_output.file_metadata.name


# ================================================================
# THOTH DECISION (DOMAIN EVENT)
# ================================================================
class ThothDecision(BaseModel):
    """
    An immutable Domain Event representing a specific conclusion reached by Thoth.

    This event captures the 'Why' (reason) and the 'What' (action), serving
    as the primary input for the Decision Ledger and Neo4j reflection.
    """

    context: DecisionContext
    action: ThothAction
    reason: str

    next_strategy: Optional[GlypharStrategy] = None
    target_pages: Optional[List[int]] = None
    llm_input: Optional[str] = None

    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def requires_reprocessing(self) -> bool:
        """Flag to trigger the Prefect worker for a new OCR run."""
        return self.action == ThothAction.REPROCESS

    @computed_field
    @property
    def requires_llm_correction(self) -> bool:
        """Flag to trigger the LLM text refinement node."""
        return self.action == ThothAction.CORRECT

    @computed_field
    @property
    def is_final(self) -> bool:
        """Terminal state check for LangGraph flow control."""
        return self.action.is_terminal

    def to_state_dict(self) -> Dict[str, Any]:
        """
        Serializes the decision into a Projection suitable for
        LangGraph state persistence and audit logging.
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
    An aggregate of the Agent's reasoning trajectory for a specific document.

    Tracks the evolution of confidence and strategy shifts, providing the
    full story of an extraction job.
    """

    doc_hash: HashSHA256
    decisions: List[ThothDecision] = Field(default_factory=list)

    final_action: Optional[ThothAction] = None
    total_reprocess_attempts: int = 0
    final_confidence: Optional[float] = None

    model_config = ConfigDict(frozen=True)

    def add_decision(self, decision: ThothDecision) -> "DecisionHistory":
        """
        Returns a new immutable history instance with the appended decision.

        Follows the functional state transition pattern used by LangGraph.
        """
        return DecisionHistory(
            doc_hash=self.doc_hash,
            decisions=self.decisions + [decision],
            final_action=decision.action if decision.is_final else self.final_action,
            total_reprocess_attempts=self.total_reprocess_attempts
            + (1 if decision.requires_reprocessing else 0),
            final_confidence=decision.context.quality_metrics.avg_confidence,
        )

"""
LLM Correction Models — Text Refinement Contracts.

This module defines the domain entities for the text correction workflow,
enabling Thoth to request, track, and audit LLM-based improvements.
"""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, computed_field, ConfigDict

from .common import HashSHA256, CorrectionUrgency


# ================================================================
# CORRECTION REQUEST
# ================================================================
class CorrectionRequest(BaseModel):
    """
    Domain command issued to an LLM for text refinement.

    Contains the original OCR output and the operational parameters
    for the correction model.
    """

    ocr_text: str = Field(..., description="Raw OCR text with page markers")
    confidence: float = Field(..., ge=0.0, le=100.0)
    model_name: str = Field(..., description="LLM model identifier (e.g., 'claude-3-5')")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8000, ge=100)

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def urgency(self) -> CorrectionUrgency:
        """
        Determines correction urgency based on OCR confidence thresholds.
        Aligned with ThothDecisionPolicy logic.
        """
        if self.confidence < 70.0:
            return CorrectionUrgency.HIGH
        if self.confidence < 88.0:
            return CorrectionUrgency.MODERATE
        return CorrectionUrgency.LOW


# ================================================================
# CORRECTION RESPONSE
# ================================================================
class CorrectionResponse(BaseModel):
    """
    Data structure representing the output of an LLM correction task.
    """

    corrected_text: str
    model_name: str
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    processing_time_s: float = Field(..., ge=0.0)
    corrected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def tokens_per_second(self) -> float:
        """Throughput performance metric for the LLM provider."""
        if self.processing_time_s <= 0:
            return 0.0
        return self.total_tokens / self.processing_time_s


# ================================================================
# CORRECTION RECORD (AUDIT)
# ================================================================
class CorrectionRecord(BaseModel):
    """
    Immutable audit record of a successful correction event.

    Used to populate the ThothLedger and for Neo4j causal reflection.
    """

    doc_hash: HashSHA256
    doc_name: str

    original_confidence: float = Field(..., ge=0.0, le=100.0)

    original_text_hash: HashSHA256
    corrected_text_hash: HashSHA256

    model_name: str

    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    processing_time_s: float = Field(..., ge=0.0)

    corrected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    success: bool
    error_message: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def was_fallback(self) -> bool:
        """
        True if the corrected text is identical to the original,
        indicating the LLM made no changes or hit a safety bypass.
        """
        return self.original_text_hash == self.corrected_text_hash


# ================================================================
# CORRECTION METADATA (STATE SUPPORT)
# ================================================================
class CorrectionMetadata(BaseModel):
    """
    Operational metadata for tracking the correction step inside LangGraph.

    Unlike CorrectionRecord, this is mutable state for the current run.
    """

    model_name: str
    urgency: CorrectionUrgency

    attempt_number: int = Field(..., ge=0)

    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    success: Optional[bool] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def is_completed(self) -> bool:
        """Boolean check for workflow progression."""
        return self.completed_at is not None

    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """Total execution time for the correction step."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

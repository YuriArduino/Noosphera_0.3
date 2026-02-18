"""
LLM Correction Models.

Domain models for text correction workflow.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, computed_field

from .common import HashSHA256, CorrectionUrgency


# ================================================================
# CORRECTION REQUEST
# ================================================================
class CorrectionRequest(BaseModel):
    """
    Domain command sent to LLM for text correction.
    """

    ocr_text: str = Field(..., description="OCR text with page markers")
    confidence: float = Field(..., ge=0.0, le=100.0)
    model_name: str = Field(..., description="LLM model identifier")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8000, ge=100)

    model_config = {"frozen": True}

    @computed_field
    @property
    def urgency(self) -> CorrectionUrgency:
        """
        Determine correction urgency based on OCR confidence.
        """
        if self.confidence < 70.0:
            return CorrectionUrgency.HIGH
        if self.confidence < 85.0:
            return CorrectionUrgency.MODERATE
        return CorrectionUrgency.LOW


# ================================================================
# CORRECTION RESPONSE
# ================================================================
class CorrectionResponse(BaseModel):
    """
    LLM correction result.
    """

    corrected_text: str
    model_name: str
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    processing_time_s: float = Field(..., ge=0.0)
    corrected_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": True}

    @computed_field
    @property
    def tokens_per_second(self) -> float:
        """
        Throughput metric.
        """
        if self.processing_time_s <= 0:
            return 0.0
        return self.total_tokens / self.processing_time_s


# ================================================================
# CORRECTION RECORD (AUDIT)
# ================================================================
class CorrectionRecord(BaseModel):
    """
    Immutable audit record of a correction event.
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

    corrected_at: datetime = Field(default_factory=datetime.utcnow)

    success: bool
    error_message: Optional[str] = None

    model_config = {"frozen": True}

    @computed_field
    @property
    def was_fallback(self) -> bool:
        """
        True if corrected text equals original text.
        """
        return self.original_text_hash == self.corrected_text_hash


# ================================================================
# CORRECTION METADATA (STATE SUPPORT)
# ================================================================
class CorrectionMetadata(BaseModel):
    """
    Runtime metadata for correction step inside Thoth workflow.

    Used in ThothState to track correction progress.
    Not an audit record — purely operational state.
    """

    model_name: str
    urgency: CorrectionUrgency

    attempt_number: int = Field(..., ge=0)

    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    success: Optional[bool] = None
    error_message: Optional[str] = None

    model_config = {"frozen": True}

    @computed_field
    @property
    def is_completed(self) -> bool:
        """Correction step is completed if completed_at is set."""
        return self.completed_at is not None

    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of correction step in seconds, or None if not completed."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

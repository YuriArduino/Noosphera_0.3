"""
Decision thresholds configuration for Thoth Agent.

Confidence thresholds for OCR quality assessment and action triggers.
"""

from pydantic import Field, model_validator

from .base import ThothBaseSettings


class ThresholdSettings(ThothBaseSettings):
    """
    Configuration for decision thresholds.

    All thresholds are confidence percentages (0.0 - 100.0).

    Threshold hierarchy (must satisfy):
        CRITICAL <= MIN_ACCEPT <= REPROCESS <= LLM_CORRECTION

    Example:
        >>> from thoth.config import threshold_settings
        >>> if confidence < threshold_settings.REPROCESS_THRESHOLD:
        ...     action = "reprocess"
    """

    # ---------------------------------------------------------------
    # CONFIDENCE THRESHOLDS
    # ---------------------------------------------------------------
    LLM_CORRECTION_THRESHOLD: float = Field(
        default=92.0,
        ge=0.0,
        le=100.0,
        description="Confidence threshold for LLM correction",
    )

    REPROCESS_THRESHOLD: float = Field(
        default=88.0,
        ge=0.0,
        le=100.0,
        description="Confidence threshold for reprocessing",
    )

    MIN_CONFIDENCE_ACCEPT: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        description="Minimum acceptable confidence without action",
    )

    CRITICAL_QUALITY_THRESHOLD: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Below this → manual review required",
    )

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------
    @model_validator(mode="after")
    def validate_threshold_order(self):
        """
        Ensure thresholds maintain logical order.

        Required order:
            CRITICAL <= MIN_ACCEPT <= REPROCESS <= LLM_CORRECTION
        """
        if not (
            self.CRITICAL_QUALITY_THRESHOLD
            <= self.MIN_CONFIDENCE_ACCEPT
            <= self.REPROCESS_THRESHOLD
            <= self.LLM_CORRECTION_THRESHOLD
        ):
            raise ValueError(
                "Thresholds must satisfy: "
                f"CRITICAL ({self.CRITICAL_QUALITY_THRESHOLD}) <= "
                f"MIN_ACCEPT ({self.MIN_CONFIDENCE_ACCEPT}) <= "
                f"REPROCESS ({self.REPROCESS_THRESHOLD}) <= "
                f"LLM_CORRECTION ({self.LLM_CORRECTION_THRESHOLD})"
            )
        return self

    # ---------------------------------------------------------------
    # HELPER METHODS
    # ---------------------------------------------------------------
    def get_action(self, confidence: float) -> str:
        """
        Determine action based on confidence score.

        Args:
            confidence: OCR confidence percentage (0-100)

        Returns:
            Action string: "reject", "reprocess", "correct", or "approve"
        """
        if confidence < self.CRITICAL_QUALITY_THRESHOLD:
            return "reject"
        elif confidence < self.REPROCESS_THRESHOLD:
            return "reprocess"
        elif confidence < self.LLM_CORRECTION_THRESHOLD:
            return "correct"
        else:
            return "approve"

    def needs_action(self, confidence: float) -> bool:
        """Check if confidence requires any action."""
        return confidence < self.LLM_CORRECTION_THRESHOLD


# ================================================================
# GLOBAL INSTANCE
# ================================================================
threshold_settings = ThresholdSettings()

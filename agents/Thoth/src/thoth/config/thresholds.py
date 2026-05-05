"""
Decision Thresholds Configuration — Thoth Agent.

Defines the confidence boundaries used to trigger specific Agent actions
such as LLM correction, reprocessing, or human escalation.
"""

from pydantic import Field, model_validator, ConfigDict
from .base import ThothBaseSettings
from ..domain.common import ThothAction


class ThresholdSettings(ThothBaseSettings):
    """
    Configuration for OCR quality decision boundaries.

    All values represent confidence percentages (0.0 to 100.0).
    These values are the internal representation of the doctrine defined
    in Glyphar's analysis.yaml.
    """

    # ---------------------------------------------------------------
    # CONFIDENCE THRESHOLDS
    # ---------------------------------------------------------------

    # Above this: OCR is near-perfect, no action needed.
    # Below this: LLM correction is recommended to reach 99%+ fidelity.
    ACCEPTANCE_CEILING: float = Field(
        default=90.0,
        ge=0.0,
        le=100.0,
        description="Confidence threshold for automatic approval without correction",
    )

    # Above this: Text is clear enough for the LLM to refine safely.
    # Below this: OCR is too garbled; LLM correction might hallucinate.
    CORRECTION_FLOOR: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Minimum confidence required for reliable LLM correction",
    )

    # Below this: The document is likely illegible or severely corrupted.
    # Reprocessing with aggressive strategies is bypassed in favor of human review.
    CRITICAL_QUALITY_LIMIT: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Below this threshold, the result is flagged as critical/unusable",
    )

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------
    @model_validator(mode="after")
    def validate_threshold_order(self) -> "ThresholdSettings":
        """
        Ensures that thresholds follow a logical progression:
        CRITICAL < CORRECTION < ACCEPTANCE
        """
        if not (self.CRITICAL_QUALITY_LIMIT <= self.CORRECTION_FLOOR <= self.ACCEPTANCE_CEILING):
            raise ValueError(
                "Threshold hierarchy violation. Required: "
                f"CRITICAL ({self.CRITICAL_QUALITY_LIMIT}) <= "
                f"CORRECTION ({self.CORRECTION_FLOOR}) <= "
                f"ACCEPTANCE ({self.ACCEPTANCE_CEILING})"
            )
        return self

    # ---------------------------------------------------------------
    # HEURISTIC HELPERS
    # ---------------------------------------------------------------
    def get_recommended_action(self, confidence: float, attempt: int = 1) -> ThothAction:
        """
        Maps a confidence score to a formal ThothAction.

        Logic:
            1. < CRITICAL: ESCALATE (Human needed)
            2. < CORRECTION: REPROCESS (Try better OCR)
            3. < ACCEPTANCE: CORRECT (Refine with LLM)
            4. >= ACCEPTANCE: ACCEPT (Done)
        """
        if confidence < self.CRITICAL_QUALITY_LIMIT:
            return ThothAction.ESCALATE

        if confidence < self.CORRECTION_FLOOR:
            # If we already tried reprocessing, escalate instead of looping forever
            return ThothAction.REPROCESS if attempt < 2 else ThothAction.ESCALATE

        if confidence < self.ACCEPTANCE_CEILING:
            return ThothAction.CORRECT

        return ThothAction.ACCEPT

    def requires_intervention(self, confidence: float) -> bool:
        """Determines if the document requires any step beyond immediate approval."""
        return confidence < self.ACCEPTANCE_CEILING

    model_config = ConfigDict(frozen=True)


# ================================================================
# GLOBAL INSTANCE
# ================================================================
threshold_settings = ThresholdSettings()

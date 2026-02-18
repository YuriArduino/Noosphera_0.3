"""
Decision Policy — Thoth Agent Autonomy Logic.
"""

from .decision import (
    DecisionContext,
    ThothDecision,
)
from .common import ThothAction, GlypharStrategy


class ThothDecisionPolicy:
    """
    Determines the next action for Thoth based on OCR quality metrics.
    """

    # Threshold configuration (tunable during experimentation phase)
    MIN_ACCEPTABLE_CONFIDENCE = 85.0
    MIN_CORRECTABLE_CONFIDENCE = 70.0
    MAX_REPROCESS_ATTEMPTS = 2

    @classmethod
    def evaluate(cls, context: DecisionContext) -> ThothDecision:
        """
        Evaluate current OCR output and decide next action.
        """

        metrics = context.quality_metrics
        avg_conf = metrics.avg_confidence
        poor_pages = metrics.poor_pages_count
        attempts = context.attempt_number

        # ------------------------------------------------------------
        # 1️⃣ Accept if quality is excellent
        # ------------------------------------------------------------
        if avg_conf >= cls.MIN_ACCEPTABLE_CONFIDENCE and poor_pages == 0:
            return ThothDecision(
                context=context,
                action=ThothAction.ACCEPT,
                reason=(f"High OCR confidence ({avg_conf:.2f}%) " "with no poor pages detected."),
            )

        # ------------------------------------------------------------
        # 2️⃣ Reprocess if quality is low and attempts available
        # ------------------------------------------------------------
        if avg_conf < cls.MIN_CORRECTABLE_CONFIDENCE:
            if attempts < cls.MAX_REPROCESS_ATTEMPTS:
                return ThothDecision(
                    context=context,
                    action=ThothAction.REPROCESS,
                    reason=(
                        f"Low average confidence ({avg_conf:.2f}%). "
                        "Reprocessing with alternative strategy."
                    ),
                    next_strategy=GlypharStrategy.AGGRESSIVE,
                )

        # ------------------------------------------------------------
        # 3️⃣ LLM correction if moderate confidence
        # ------------------------------------------------------------
        if cls.MIN_CORRECTABLE_CONFIDENCE <= avg_conf < cls.MIN_ACCEPTABLE_CONFIDENCE:
            return ThothDecision(
                context=context,
                action=ThothAction.CORRECT,
                reason=(
                    f"Moderate confidence ({avg_conf:.2f}%). " "Applying LLM-based correction."
                ),
            )

        # ------------------------------------------------------------
        # 4️⃣ Escalate to human review (HITL fallback)
        # ------------------------------------------------------------
        return ThothDecision(
            context=context,
            action=ThothAction.ESCALATE,
            reason=("Maximum reprocessing attempts reached or " "confidence remains insufficient."),
        )

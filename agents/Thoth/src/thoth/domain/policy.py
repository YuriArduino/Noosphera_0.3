"""
Decision Policy — Thoth Agent Autonomy Logic.

This module encapsulates the business rules and heuristics used by the Agent
to interpret OCR quality metrics and select the next operational state.
"""

from .decision import (
    DecisionContext,
    ThothDecision,
)
from .common import ThothAction, GlypharStrategy


class ThothDecisionPolicy:
    """
    Heuristic engine that determines the next step in the extraction lifecycle.

    This policy acts as the 'Controlled Freedom' guardrail, ensuring Thoth
    follows the Noosphera quality doctrine while deciding between reprocessing,
    LLM correction, or human escalation.
    """

    # --- Doctrine Thresholds (To be dynamically injected via YAML in future phases) ---
    MIN_ACCEPTABLE_CONFIDENCE = 88.0  # Threshold for direct approval
    MIN_CORRECTABLE_CONFIDENCE = 70.0  # Below this, LLM correction is unreliable
    MAX_REPROCESS_ATTEMPTS = 2

    @classmethod
    def evaluate(cls, context: DecisionContext) -> ThothDecision:
        """
        Evaluates the current OCR context and returns a formal ThothDecision.
        """

        metrics = context.quality_metrics
        avg_conf = metrics.avg_confidence
        poor_pages = metrics.poor_pages_count
        attempts = context.attempt_number

        # ------------------------------------------------------------
        # 1. APPROVAL: High confidence and no critical quality issues
        # ------------------------------------------------------------
        if avg_conf >= cls.MIN_ACCEPTABLE_CONFIDENCE and poor_pages == 0:
            return ThothDecision(
                context=context,
                action=ThothAction.ACCEPT,
                reason=(
                    f"OCR result approved: High confidence ({avg_conf:.2f}%) "
                    "with no poor quality pages detected."
                ),
            )

        # ------------------------------------------------------------
        # 2. REPROCESS: Quality is too low for reliable LLM correction
        # ------------------------------------------------------------
        if avg_conf < cls.MIN_CORRECTABLE_CONFIDENCE:
            if attempts < cls.MAX_REPROCESS_ATTEMPTS:
                return ThothDecision(
                    context=context,
                    action=ThothAction.REPROCESS,
                    reason=(
                        f"Low confidence ({avg_conf:.2f}%). "
                        "Attempting recovery via aggressive preprocessing strategy."
                    ),
                    next_strategy=GlypharStrategy.AGGRESSIVE,
                )

        # ------------------------------------------------------------
        # 3. CORRECT: Moderate confidence allows for LLM refinement
        # ------------------------------------------------------------
        if cls.MIN_CORRECTABLE_CONFIDENCE <= avg_conf < cls.MIN_ACCEPTABLE_CONFIDENCE:
            return ThothDecision(
                context=context,
                action=ThothAction.CORRECT,
                reason=(
                    f"Moderate confidence ({avg_conf:.2f}%). "
                    "Deploying LLM correction to refine text fidelity."
                ),
            )

        # ------------------------------------------------------------
        # 4. ESCALATE: HITL fallback when automation fails
        # ------------------------------------------------------------
        return ThothDecision(
            context=context,
            action=ThothAction.ESCALATE,
            reason=(
                "Automated recovery exhausted. Confidence remains below "
                f"threshold ({avg_conf:.2f}%) or critical errors persist."
            ),
        )

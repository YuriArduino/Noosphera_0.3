"""
Common types and enums for Thoth Domain Model.
Shared across all domain entities for consistency.
"""

from enum import Enum
from typing import Annotated, Tuple
from pydantic import StringConstraints


# ================================================================
# QUALITY ENUMS
# ================================================================
class PageQuality(str, Enum):
    """
    Page quality classification as produced by Glyphar OCR.
    Must match OCR JSON output exactly.
    """

    EXCELLENT = "excellent"
    FAIR = "fair"
    POOR = "poor"

    @property
    def is_acceptable(self) -> bool:
        """Acceptable quality for final output."""
        return self in {PageQuality.EXCELLENT, PageQuality.FAIR}

    @property
    def is_critical(self) -> bool:
        """Critical quality level that may trigger reprocessing."""
        return self == PageQuality.POOR


# ================================================================
# LAYOUT
# ================================================================
class LayoutType(str, Enum):
    """Document layout structure."""

    SINGLE = "single"
    DOUBLE = "double"
    MULTI = "multi"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


# ================================================================
# GLYPHAR STRATEGIES
# ================================================================
class GlypharStrategy(str, Enum):
    """
    OCR processing strategies.

    Ordered from fastest to most aggressive.
    """

    FAST = "fast_scan"  # Minimal preprocessing
    BALANCED = "high_accuracy"  # Default strategy
    AGGRESSIVE = "noisy_documents"  # Heavy preprocessing

    @property
    def is_aggressive(self) -> bool:
        """Indicates if this strategy is the most aggressive option."""
        return self == GlypharStrategy.AGGRESSIVE


# ================================================================
# THOTH ACTIONS
# ================================================================
class ThothAction(str, Enum):
    """
    Actions Thoth can take after assessing OCR results.
    """

    ACCEPT = "accept"  # Final, approved result
    CORRECT = "correct"  # LLM correction required
    REPROCESS = "reprocess"  # Retry OCR with different strategy
    ESCALATE = "escalate"  # Human-in-the-loop

    @property
    def is_terminal(self) -> bool:
        """Indicates if this action is a final decision
        with no further processing."""
        return self in {
            ThothAction.ACCEPT,
            ThothAction.ESCALATE,
        }


# ================================================================
# EXECUTION STATES
# ================================================================
class ExecutionStep(str, Enum):
    """Current step in Thoth execution graph."""

    INGEST = "ingest"
    ASSESS = "assess"
    DECIDE = "decide"
    REPROCESS = "reprocess"
    CORRECT = "correct"
    FINALIZE = "finalize"


# ================================================================
# TYPE ALIASES
# ================================================================
HashSHA256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]

DocumentID = Annotated[str, StringConstraints(min_length=1, max_length=256)]

PageID = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+_\d{8}_\d{3}$")]

BoundingBox = Tuple[int, int, int, int]


# ================================================================
# CORRECTION
# ================================================================
class CorrectionUrgency(str, Enum):
    """Urgency level for LLM correction based on OCR confidence."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

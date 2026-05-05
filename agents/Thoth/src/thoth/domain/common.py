"""
Common types and enums for the Thoth Domain Model.

This module anchors the shared language between the Agent and the Glyphar engine,
re-exporting core types to maintain the Single Source of Truth (SST) while
adding agent-specific decision logic.
"""

from enum import Enum
from typing import Annotated, Tuple
from pydantic import StringConstraints

# --- Re-exporting SST from Glyphar Core ---
from glyphar.core.types import PageQuality as GlypharPageQuality
from glyphar.core.types import LayoutType as GlypharLayoutType


# ================================================================
# QUALITY ENUMS (Extended with Business Logic)
# ================================================================
class PageQuality(str, Enum):
    """
    Page quality classification mapped from Glyphar OCR.
    Includes business logic properties for Agent decision-making.
    """

    EXCELLENT = GlypharPageQuality.EXCELLENT
    GOOD = GlypharPageQuality.GOOD
    FAIR = GlypharPageQuality.FAIR
    POOR = GlypharPageQuality.POOR
    UNKNOWN = GlypharPageQuality.UNKNOWN

    @property
    def is_acceptable(self) -> bool:
        """Determines if quality is high enough to bypass heavy correction."""
        return self in {PageQuality.EXCELLENT, PageQuality.GOOD, PageQuality.FAIR}

    @property
    def is_critical(self) -> bool:
        """Critical quality level that triggers strategy re-evaluation."""
        return self == PageQuality.POOR


# ================================================================
# LAYOUT (SST Mapping)
# ================================================================
class LayoutType(str, Enum):
    """Document layout structure mapped from Glyphar."""

    SINGLE = GlypharLayoutType.SINGLE
    DOUBLE = GlypharLayoutType.DOUBLE
    MULTI = GlypharLayoutType.MULTI
    COMPLEX = GlypharLayoutType.COMPLEX
    UNKNOWN = GlypharLayoutType.UNKNOWN


# ================================================================
# GLYPHAR STRATEGIES (Mapping to YAML Doctrine)
# ================================================================
class GlypharStrategy(str, Enum):
    """
    OCR processing strategies defined in Glyphar's strategies/*.yaml.
    Ordered from fastest to most aggressive.
    """

    FAST = "fast_scan"  # Minimal preprocessing
    BALANCED = "high_accuracy"  # Standard balance
    AGGRESSIVE = "noisy_documents"  # Full preprocessing stack
    CUSTOM = "custom"  # Agent-generated experiment

    @property
    def is_aggressive(self) -> bool:
        """Indicates if this strategy uses maximum resources."""
        return self == GlypharStrategy.AGGRESSIVE


# ================================================================
# THOTH ACTIONS (Agent Intentions)
# ================================================================
class ThothAction(str, Enum):
    """Actions Thoth can take after assessing OCR results."""

    ACCEPT = "accept"  # Final, approved result
    CORRECT = "correct"  # LLM-based text refinement required
    REPROCESS = "reprocess"  # Retry OCR with a different strategy
    ESCALATE = "escalate"  # Human-in-the-loop required

    @property
    def is_terminal(self) -> bool:
        """Indicates if this action ends the current execution cycle."""
        return self in {ThothAction.ACCEPT, ThothAction.ESCALATE}


# ================================================================
# EXECUTION STATES (LangGraph Nodes)
# ================================================================
class ExecutionStep(str, Enum):
    """Current step/node in the Thoth execution graph."""

    INGEST = "ingest"
    ASSESS = "assess"
    DECIDE = "decide"
    REPROCESS = "reprocess"
    CORRECT = "correct"
    FINALIZE = "finalize"


# ================================================================
# TYPE ALIASES (Strict Validation)
# ================================================================

# Standard hex-encoded SHA256
HashSHA256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]

# Logical document identifier
DocumentID = Annotated[str, StringConstraints(min_length=1, max_length=256)]

# Physical page ID: prefix_YYYYMMDD_001
PageID = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+_\d{8}_\d{3}$")]

# Geometric representation: (left, top, width, height)
BoundingBox = Tuple[int, int, int, int]


# ================================================================
# CORRECTION (LLM Layer)
# ================================================================
class CorrectionUrgency(str, Enum):
    """Urgency level for LLM correction based on OCR confidence drops."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

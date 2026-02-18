"""
Common types and enums for Thoth Domain Model.

Shared across all domain entities for consistency.
"""

from enum import Enum
from typing import Annotated
from pydantic import Field, StringConstraints


# ================================================================
# QUALITY ENUMS (from Glyphar analysis.yaml)
# ================================================================
class PageQuality(str, Enum):
    """
    Page quality classification based on sharpness and contrast.

    Source: docs/capabilities/analysis.yaml
    """

    EXCELLENT = "excellent"  # sharpness > 250, contrast > 0.6, accuracy 92-95%
    GOOD = "good"  # sharpness > 150, contrast > 0.4, accuracy 88-92%
    FAIR = "fair"  # sharpness > 80, contrast > 0.25, accuracy 82-88%
    POOR = "poor"  # sharpness < 80, contrast < 0.25, accuracy 75-82%


class LayoutType(str, Enum):
    """Document layout structure."""

    SINGLE = "single"  # Single column (95% of documents)
    DOUBLE = "double"  # Two columns
    MULTI = "multi"  # Multiple columns
    COMPLEX = "complex"  # Irregular layout
    UNKNOWN = "unknown"  # Could not detect


# ================================================================
# GLYPHAR STRATEGIES (from docs/strategies/*.yaml)
# ================================================================
class GlypharStrategy(str, Enum):
    """
    OCR processing strategies.

    Source: docs/strategies/
    """

    FAST_SCAN = "fast_scan"  # Speed > accuracy (1.5s/pág, 85-90%)
    HIGH_ACCURACY = "high_accuracy"  # Accuracy > speed (2.8s/pág, 90-95%)
    NOISY_DOCUMENTS = "noisy_documents"  # Robustness (3.5s/pág, 82-90%)


# ================================================================
# THOTH ACTIONS (Agent decision types)
# ================================================================
class ThothAction(str, Enum):
    """Actions Thoth can take after assessing OCR results."""

    APPROVE = "approve"  # Confidence >= 92%, no action needed
    CORRECT = "correct"  # 88% <= confidence < 92%, send to LLM
    REPROCESS = "reprocess"  # confidence < 88% or pages "poor"
    REJECT = "reject"  # confidence < 50%, manual review


# ================================================================
# EXECUTION STATES (LangGraph workflow)
# ================================================================
class ExecutionStep(str, Enum):
    """Current step in Thoth's execution graph."""

    INGEST = "ingest"
    ASSESS = "assess"
    DECIDE = "decide"
    REPROCESS = "reprocess"
    CORRECT = "correct"
    FINALIZE = "finalize"


# ================================================================
# TYPE ALIASES
# ================================================================
# SHA256 hash strings
HashSHA256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]

# Document ID (uuid4 or content_hash)
DocumentID = Annotated[str, StringConstraints(min_length=1, max_length=256)]

# Page ID format: {doc_prefix}_{date}_{page_number:03d}
PageID = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+_\d{8}_\d{3}$")]

# Bounding box coordinates (left, top, width, height)
BoundingBox = tuple[int, int, int, int]

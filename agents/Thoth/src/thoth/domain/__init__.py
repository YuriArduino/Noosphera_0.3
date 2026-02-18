"""
Thoth Domain Model — Pure Business Entities.

Pydantic V2 models representing core business concepts.
No infrastructure dependencies — pure domain logic.
"""

# ================================================================
# COMMON
# ================================================================
from .common import (
    PageQuality,
    LayoutType,
    GlypharStrategy,
    ThothAction,
    ExecutionStep,
    CorrectionUrgency,
    HashSHA256,
    PageID,
    DocumentID,
    BoundingBox,
)

# ================================================================
# OCR
# ================================================================
from .ocr import (
    FileMetadata,
    ColumnResult,
    PageResult,
    OCRStatistics,
    OCRConfig,
    ProcessingMetadata,
    OCROutput,
)

# ================================================================
# STATE (execution layer)
# ================================================================
from .state import (
    ThothState,
    DecisionProjection,
    CorrectionProjection,
    ExecutionMetadata,
)

# ================================================================
# DECISION (domain layer)
# ================================================================
from .decision import (
    QualityMetrics,
    DecisionContext,
    ThothDecision,
    DecisionHistory,
)

# ================================================================
# CORRECTION (domain layer)
# ================================================================
from .correction import (
    CorrectionRequest,
    CorrectionResponse,
    CorrectionRecord,
    CorrectionMetadata,
)

__all__ = [
    # Common
    "PageQuality",
    "LayoutType",
    "GlypharStrategy",
    "ThothAction",
    "ExecutionStep",
    "CorrectionUrgency",
    "HashSHA256",
    "PageID",
    "DocumentID",
    "BoundingBox",
    # OCR
    "FileMetadata",
    "ColumnResult",
    "PageResult",
    "OCRStatistics",
    "OCRConfig",
    "ProcessingMetadata",
    "OCROutput",
    # State
    "ThothState",
    "DecisionProjection",
    "CorrectionProjection",
    "ExecutionMetadata",
    # Decision
    "QualityMetrics",
    "DecisionContext",
    "ThothDecision",
    "DecisionHistory",
    # Correction
    "CorrectionRequest",
    "CorrectionResponse",
    "CorrectionRecord",
    "CorrectionMetadata",
]

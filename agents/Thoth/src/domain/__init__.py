"""
Thoth Domain Model — Pure Business Entities.

Pydantic V2 models representing core business concepts.
No infrastructure dependencies — pure domain logic.

Usage:
    >>> from thoth.domain import OCROutput, ThothState, ThothDecision
    >>> from thoth.domain.common import PageQuality, ThothAction

Submodules:
    - common: Enums and shared types
    - ocr: OCR output models (Glyphar integration)
    - state: LangGraph state models
    - decision: Agent decision models
    - correction: LLM correction models
"""

from .common import (
    PageQuality,
    LayoutType,
    GlypharStrategy,
    ThothAction,
    ExecutionStep,
    HashSHA256,
    PageID,
    DocumentID,
    BoundingBox,
)

from .ocr import (
    FileMetadata,
    ColumnResult,
    PageResult,
    OCRStatistics,
    OCRConfig,
    ProcessingMetadata,
    OCROutput,
)

from .state import (
    ThothState,
    DecisionRecord,
    CorrectionMetadata,
    ExecutionMetadata,
)

from .decision import (
    QualityMetrics,
    DecisionContext,
    ThothDecision,
    DecisionHistory,
)

from .correction import (
    CorrectionRequest,
    CorrectionResponse,
    CorrectionRecord,
)

__all__ = [
    # Common
    "PageQuality",
    "LayoutType",
    "GlypharStrategy",
    "ThothAction",
    "ExecutionStep",
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
    "DecisionRecord",
    "CorrectionMetadata",
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
]

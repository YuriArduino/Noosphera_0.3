"""
Thoth State Model — LangGraph TypedDict.

Execution state that flows through Thoth agent graph.
Pure orchestration state (no domain logic).
All domain objects must be serialized before entering state.
"""

from typing import TypedDict, List, Optional, Dict

from .ocr import OCROutput
from .common import GlypharStrategy, ExecutionStep


# ================================================================
# DECISION PROJECTION (STATE VIEW)
# ================================================================
class DecisionProjection(TypedDict, total=False):
    """
    Serialized projection of ThothDecision for execution tracking.

    Derived from ThothDecision.to_state_dict().
    """

    doc_hash: str
    doc_name: str
    action: str
    reason: str
    metrics: Dict[str, float | int | str]
    target_pages: Optional[List[int]]
    current_strategy: Optional[str]
    next_strategy: Optional[str]
    llm_input: Optional[str]


# ================================================================
# CORRECTION PROJECTION (STATE VIEW)
# ================================================================
class CorrectionProjection(TypedDict, total=False):
    """
    Serialized projection of CorrectionRecord.
    """

    doc_hash: str
    doc_name: str
    model_name: str
    original_confidence: float
    prompt_tokens: int
    completion_tokens: int
    processing_time_s: float
    success: bool
    error_message: Optional[str]
    corrected_at: str


# ================================================================
# EXECUTION METADATA
# ================================================================
class ExecutionMetadata(TypedDict, total=False):
    """
    Global execution tracking information.
    """

    ingest_timestamp: str
    finalize_timestamp: str
    total_documents: int
    total_errors: int
    duration_seconds: Optional[float]


# ================================================================
# THOTH STATE
# ================================================================
class ThothState(TypedDict):
    """
    Complete execution state for Thoth LangGraph agent.

    Flow:
        ingest → assess → decide → {reprocess | correct} → finalize
    """

    # === INPUT ===
    documents: List[str]
    initial_strategy: GlypharStrategy

    # === PERCEPTION ===
    ocr_results: List[OCROutput]

    # === DECISION MEMORY ===
    decisions: List[DecisionProjection]
    reprocess_attempts: Dict[str, int]  # doc_hash → attempts
    max_reprocess_attempts: int

    # === LLM CORRECTIONS ===
    llm_corrections: Dict[str, CorrectionProjection]  # doc_hash → correction

    # === OUTPUT ===
    approved_results: List[OCROutput]
    errors: List[Dict[str, str]]

    # === CONTROL ===
    current_step: ExecutionStep
    stop_execution: bool

    # === META ===
    execution_meta: ExecutionMetadata

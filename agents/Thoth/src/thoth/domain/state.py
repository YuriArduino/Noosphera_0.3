"""
Thoth State Model — LangGraph TypedDict.

This module defines the execution state that flows through the Thoth agent
graph. It separates domain-rich objects (Perception) from serialized snapshots
(Projections) for optimized orchestration and persistence.

Design Principles:
    - Immutability: State is updated via LangGraph transitions.
    - Traceability: Projections provide a historical log of decisions.
    - Decoupling: Execution metadata is kept separate from domain data.
"""

from typing import TypedDict, List, Optional, Dict, Union
from .ocr import OCROutput
from .common import GlypharStrategy, ExecutionStep


# ================================================================
# DECISION PROJECTION (STATE VIEW)
# ================================================================
class DecisionProjection(TypedDict, total=False):
    """
    Serialized snapshot of a ThothDecision for execution tracking.

    Used to record the rationale behind strategy changes or approval steps.
    """

    doc_hash: str
    doc_name: str
    action: str
    reason: str
    metrics: Dict[str, Union[float, int, str]]
    target_pages: Optional[List[int]]
    current_strategy: Optional[str]
    next_strategy: Optional[str]
    llm_input: Optional[str]


# ================================================================
# CORRECTION PROJECTION (STATE VIEW)
# ================================================================
class CorrectionProjection(TypedDict, total=False):
    """
    Serialized snapshot of an LLM text refinement record.
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
    Metadata for global execution tracking and performance auditing.
    """

    ingest_timestamp: str
    finalize_timestamp: str
    total_documents: int
    total_errors: int
    duration_seconds: Optional[float]
    # Added batch_id to align with SST persistence logic
    batch_id: Optional[str]


# ================================================================
# THOTH STATE (CORE CONTRACT)
# ================================================================
class ThothState(TypedDict):
    """
    Complete execution state for the Thoth LangGraph agent.

    This state is persisted by the PostgresSaver (Checkpointer) to allow
    resuming workflows and iterative experimentation.

    Flow Sequence:
        ingest -> triage -> ocr -> analysis -> decide
        -> {reprocess | correct} -> finalize -> memory_reflection
    """

    # === INPUT ===
    documents: List[str]  # List of file paths or identifiers
    initial_strategy: GlypharStrategy

    # === PERCEPTION (Domain Objects) ===
    # These contain the full SQLModel-based OCR results
    ocr_results: List[OCROutput]

    # === DECISION MEMORY (Operational Tracking) ===
    decisions: List[DecisionProjection]
    reprocess_attempts: Dict[str, int]
    max_reprocess_attempts: int

    # === LLM CORRECTIONS ===
    llm_corrections: Dict[str, CorrectionProjection]

    # === MEMORY CONTEXT (Cognitive Layer) ===
    # Data for the MemoryManager/LangMem reflection loop
    memory_summary_version: Optional[int]
    memory_window_ids: List[str]
    memory_reflection_required: bool
    memory_reflection_performed: bool
    memory_influence_notes: Optional[str]

    # === OUTPUT ===
    approved_results: List[OCROutput]
    errors: List[Dict[str, str]]

    # === CONTROL ===
    current_step: ExecutionStep
    stop_execution: bool

    # === META ===
    execution_meta: ExecutionMetadata

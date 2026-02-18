"""
Thoth State Model — LangGraph TypedDict.

State that flows through the agent's execution graph.
"""

from typing import TypedDict, List, Optional, Literal
from .ocr import OCROutput, PageResult
from .common import GlypharStrategy, ExecutionStep, ThothAction


class DecisionRecord(TypedDict, total=False):
    """
    Record of a decision made by Thoth.

    Stored in state for audit trail and learning.
    """
    doc_hash: str
    doc_name: str
    action: ThothAction
    reason: str
    metrics: dict
    target_pages: Optional[List[int]]
    current_strategy: Optional[GlypharStrategy]
    next_strategy: Optional[GlypharStrategy]
    reprocess_log: Optional[dict]
    llm_input: Optional[str]


class CorrectionMetadata(TypedDict, total=False):
    """Metadata for LLM corrections applied."""
    doc_hash: str
    doc_name: str
    original_confidence: float
    corrected_at: str
    model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]


class ExecutionMetadata(TypedDict, total=False):
    """Execution tracking metadata."""
    ingest_timestamp: str
    finalize_timestamp: str
    total_documents: int
    total_errors: int
    reprocess_attempts: dict[str, int]
    duration_seconds: Optional[float]


class ThothState(TypedDict):
    """
    Complete state for Thoth agent LangGraph execution.

    Flows through nodes: ingest → assess → decide → {reprocess | correct} → finalize

    Attributes:
        documents: Input PDF paths to process
        initial_strategy: Starting Glyphar strategy
        ocr_results: Raw OCR outputs from Glyphar
        decisions: Decision records for each document
        reprocess_attempts: Count of reprocessing attempts per document
        max_reprocess_attempts: Maximum retry limit (default: 3)
        llm_inputs: Texts sent to LLM for correction
        corrected_texts: Texts after LLM correction
        corrections_meta Metadata for each correction
        approved_results: Final approved OCR outputs
        errors: Errors encountered during execution
        current_step: Current execution step
        stop_execution: Flag to halt execution early
        execution_metadata: Execution tracking data
    """
    # === INPUT ===
    documents: List[str]
    initial_strategy: GlypharStrategy

    # === GLYPHAR RESULTS ===
    ocr_results: List[OCROutput]
    processed_pages: List[PageResult]

    # === DECISION TRACKING ===
    decisions: List[DecisionRecord]
    reprocess_attempts: dict[str, int]
    max_reprocess_attempts: int

    # === LLM CORRECTION PIPELINE ===
    llm_inputs: List[str]
    corrected_texts: List[str]
    corrections_meta List[CorrectionMetadata]

    # === FINAL OUTPUT ===
    approved_results: List[OCROutput]
    errors: List[dict]

    # === CONTROL FLAGS ===
    current_step: Literal[
        "ingest", "assess", "decide", "reprocess", "correct", "finalize"
    ]
    stop_execution: bool

    # === METADATA ===
    execution_meta ExecutionMetadata

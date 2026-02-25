"""
Thoth Agent → Tool Contract (A2T)

Encapsulates all domain + infrastructure logic.
LLM must never import domain directly.
"""

from pathlib import Path
from typing import Dict, Any
import asyncio

# Domain
from thoth.domain.decision import DecisionContext
from thoth.domain.policy import ThothDecisionPolicy
from thoth.domain.common import GlypharStrategy

# Infrastructure
from thoth.infrastructure.ledger import ThothLedger
from thoth.infrastructure.memory_manager import ThothMemoryManager

# Glyphar Application Service
from glyphar import Glyphar


ledger = ThothLedger()
memory = ThothMemoryManager()
glyphar = Glyphar()


def glyphar_process_document(
    path: str,
    strategy: str = "balanced",
    attempt_number: int = 0,
) -> Dict[str, Any]:

    pdf_path = Path(path)

    if not pdf_path.exists():
        return {
            "status": "error",
            "message": "Document not found",
        }

    try:
        # ---------------------------------------------------------
        # 1️⃣ Perception (Handled entirely by Glyphar)
        # ---------------------------------------------------------
        ocr_output = glyphar.process(
            path=str(pdf_path),
            strategy=strategy,
        )

        # ---------------------------------------------------------
        # 2️⃣ Build Decision Context
        # ---------------------------------------------------------
        context = DecisionContext.from_ocr_output(
            ocr_output=ocr_output,
            current_strategy=GlypharStrategy(strategy),
            attempt_number=attempt_number,
        )

        # ---------------------------------------------------------
        # 3️⃣ Deterministic Policy Evaluation
        # ---------------------------------------------------------
        decision = ThothDecisionPolicy.evaluate(context)

        # ---------------------------------------------------------
        # 4️⃣ Ledger Logging
        # ---------------------------------------------------------
        ledger.log_decision(
            document_id=context.doc_name,
            document_hash=context.doc_hash,
            action=decision.action.value,
            strategy=context.current_strategy.value,
            avg_confidence=context.quality_metrics.avg_confidence,
            attempts=context.attempt_number,
            execution_step="agent_tool",
            hitl_triggered=(decision.action.value == "escalate"),
        )

        # ---------------------------------------------------------
        # 5️⃣ Background Learning
        # ---------------------------------------------------------
        if decision.action.value != "accept":
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    memory.process_decision(
                        document_id=context.doc_name,
                        document_hash=context.doc_hash,
                        avg_confidence=context.quality_metrics.avg_confidence,
                        action=decision.action.value,
                        strategy=context.current_strategy.value,
                        attempts=context.attempt_number,
                        hitl_triggered=False,
                    )
                )
            except RuntimeError:
                pass

        # ---------------------------------------------------------
        # 6️⃣ Clean Contract to LLM
        # ---------------------------------------------------------
        return {
            "status": "success",
            "document": context.doc_name,
            "avg_confidence": context.quality_metrics.avg_confidence,
            "poor_pages": context.quality_metrics.poor_pages_count,
            "decision_hint": decision.action.value,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }

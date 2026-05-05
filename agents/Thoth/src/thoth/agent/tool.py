"""
Thoth Agent Tools — Infrastructure Interface.

This module provides the specialized tools that the LangGraph Agent uses
to interact with the Glyphar OCR engine via Prefect orchestration.

It implements the 'Controlled Freedom' pattern:
1. Agent proposes a strategy (YAML-based).
2. Tool triggers an isolated Prefect Flow in an ephemeral container.
3. Tool retrieves validated results from the PostgreSQL SST.
4. Tool audits the outcome in the Decision Ledger.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from langchain_core.tools import tool
from prefect.client import get_client
from sqlmodel import Session  # 'select' removed as it was unused

# Configuration
from thoth.config import settings

# Domain
from thoth.domain.common import GlypharStrategy, ThothAction
from thoth.domain.decision import DecisionContext, QualityMetrics
from thoth.domain.policy import ThothDecisionPolicy

# Infrastructure
from thoth.infrastructure.ledger import ThothLedger
from thoth.infrastructure.memory_manager import ThothMemoryManager
from glyphar.database import engine
from glyphar.schema import FileTable

# Global Infrastructure Instances
ledger = ThothLedger()
memory = ThothMemoryManager()


class GlypharInfrastructureTool:
    """
    Bridge between the Thoth Agent and the distributed Glyphar workers.
    """

    @staticmethod
    async def _wait_for_prefect_result(
        deployment_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Triggers a Prefect Flow and polls for completion.
        """
        async with get_client() as client:
            deployment = await client.read_deployment_by_name(deployment_name)
            flow_run = await client.create_flow_run_from_deployment(
                deployment_id=deployment.id, parameters=parameters
            )

            while True:
                state = await client.read_flow_run(flow_run.id)
                if state.state.is_completed():
                    # Extract the result payload from the Prefect State
                    return await state.state.result().get()
                elif state.state.is_failed():
                    raise RuntimeError(f"Prefect Flow failed: {state.state.message}")

                await asyncio.sleep(2)

    @tool
    async def glyphar_ocr_task(
        self,
        file_path: str,
        strategy: GlypharStrategy = GlypharStrategy.BALANCED,
        overrides: Optional[Dict[str, Any]] = None,
        attempt: int = 1,
    ) -> Dict[str, Any]:
        """
        Executes a high-performance OCR job using an isolated ephemeral container.

        Use this tool when a document needs processing or reprocessing with a
        specific strategy (fast_scan, high_accuracy, noisy_documents).

        Args:
            file_path: Absolute path to the PDF document.
            strategy: Pre-defined doctrine strategy from the YAML collection.
            overrides: Optional YAML-style overrides for experimentation.
            attempt: Current retry count for this specific document.
        """

        agent_proposal = overrides or {}
        batch_id = f"thoth_exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        try:
            # 1. Orchestration: Trigger Ephemeral Worker
            prefect_result = await self._wait_for_prefect_result(
                deployment_name=settings.glyphar.GLYPHAR_FLOW_DEPLOYMENT,
                parameters={
                    "file_path": file_path,
                    "agent_overrides": agent_proposal,
                    "batch_id": batch_id,
                },
            )

            db_file_id = prefect_result.get("db_file_id")

            # 2. Perception: Access the Source of Truth (SST)
            with Session(engine) as session:
                db_record = session.get(FileTable, db_file_id)
                if not db_record:
                    raise ValueError(f"File ID {db_file_id} not found in PostgreSQL.")

            # 3. Assessment: Map DB Data back to Domain Entities
            # Extract statistics from JSONB column
            stats = db_record.statistics if isinstance(db_record.statistics, dict) else {}

            metrics = QualityMetrics(
                avg_confidence=stats.get("average_confidence", 0.0),
                poor_pages_count=stats.get("failed_pages", 0),
                fair_pages_count=0,  # Detail can be computed from ocr_pages if needed
                excellent_pages_count=stats.get("successful_pages", 0),
                min_confidence=0.0,
                max_confidence=100.0,
            )

            # Build the Context for the Decision Policy
            # Note: Since the DB record doesn't store OCROutput directly,
            # we use a reconstructed context for the heuristic evaluation.
            context = DecisionContext(
                ocr_output=None,  # In a real scenario, map db_record -> OCROutput
                quality_metrics=metrics,
                current_strategy=strategy,
                attempt_number=attempt,
            )

            # Evaluate Policy (Reprocess? Correct? Approve?)
            decision = ThothDecisionPolicy.evaluate(context)

            # 4. Learning & Audit: Update Ledger and Semantic Memory
            ledger.log_decision(
                document_id=db_record.name,
                document_hash=db_record.hash_sha256,
                action=decision.action.value,
                strategy=strategy.value,
                avg_confidence=metrics.avg_confidence,
                attempts=attempt,
                execution_step="infrastructure_tool",
                hitl_triggered=(decision.action == ThothAction.ESCALATE),
            )

            # Consolidate pattern in pgvector background memory
            await memory.process_decision(
                document_id=db_record.name,
                document_hash=db_record.hash_sha256,
                avg_confidence=metrics.avg_confidence,
                action=decision.action.value,
                strategy=strategy.value,
                attempts=attempt,
                hitl_triggered=(decision.action == ThothAction.ESCALATE),
            )

            # 5. Clean JSON contract for the LLM
            return {
                "status": "success",
                "db_file_id": db_file_id,
                "document_name": db_record.name,
                "confidence": round(metrics.avg_confidence, 2),
                "action_recommendation": decision.action.value,
                "rationale": decision.reason,
            }

        except (ValueError, RuntimeError) as e:
            return {"status": "error", "message": f"Infrastructure failure: {str(e)}"}

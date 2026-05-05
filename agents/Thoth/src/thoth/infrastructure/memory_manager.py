"""
Background memory management for the Thoth Agent.

Uses LangMem to automatically extract long-term semantic memories
from agent interactions and internal decision trajectories.

This layer is:
- Persistent: Uses PostgreSQL + pgvector (SST).
- Passive: Operates in background threads to consolidate patterns.
- Cognitive: Focused on learning from failures and corrections.
"""

from __future__ import annotations

from typing import List, Optional
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.store.postgres import PostgresStore  # Upgraded from InMemory
from langmem import create_memory_store_manager

from thoth.config import memory_settings
from glyphar.database import DATABASE_URL  # Using our SST connection string


class ThothMemoryManager:
    """
    Extracts semantic knowledge from Thoth's internal decisions
    and interactions using LangMem and persists it in PostgreSQL.
    """

    def __init__(self) -> None:
        self.enabled: bool = memory_settings.MEMORY_ENABLED

        if not self.enabled:
            self.store = None
            self.memory_manager = None
            return

        # -----------------------------------------------------
        # Persistent Vector Store (PostgreSQL + pgvector)
        # -----------------------------------------------------
        # This replaces InMemoryStore to ensure memory survives container restarts.
        # It creates a 'langgraph_store' table in your glyphar_db.
        self.store = PostgresStore.from_conn_string(DATABASE_URL)

        # Ensure tables exist (In Noosphera 0.3, this can also be handled by Alembic)
        self.store.setup()

        # -----------------------------------------------------
        # Reasoning LLM for Memory Reflection
        # -----------------------------------------------------
        # Used by LangMem to 'think' about the messages and extract insights.
        self.llm = init_chat_model(
            model=memory_settings.REFLECTION_MODEL,
            model_provider=memory_settings.REFLECTION_PROVIDER,
        )

        # -----------------------------------------------------
        # Background Memory Manager
        # -----------------------------------------------------
        # Manages the 'Reflection' loop. Namespace isolates Thoth from other tools.
        self.memory_manager = create_memory_store_manager(
            self.llm,
            storage=self.store,
            namespace=("thoth", "semantic_memory"),
        )

    # ==========================================================
    # INTERNAL DECISION MEMORY
    # ==========================================================

    async def process_decision(
        self,
        document_id: str,
        document_hash: str,
        avg_confidence: float,
        action: str,
        strategy: str,
        attempts: int,
        hitl_triggered: bool,
        correction_summary: Optional[str] = None,
    ) -> None:
        """
        Converts internal reasoning logs into semantic patterns.

        LangMem will analyze these messages to extract rules like:
        'If document hash starts with X and confidence is Y, strategy Z fails'.
        """
        if not self.enabled or not self.memory_manager:
            return

        messages: List[AnyMessage] = [
            SystemMessage(content="Thoth OCR agent internal reasoning trajectory."),
            HumanMessage(
                content=(
                    f"Context:\n- Doc ID: {document_id}\n"
                    f"- Hash: {document_hash}\n"
                    f"- Avg Confidence: {avg_confidence}"
                )
            ),
            AIMessage(
                content=(
                    f"Decision logic:\n- Attempted Strategy: {strategy}\n"
                    f"- Final Action: {action}\n- Total Attempts: {attempts}\n"
                    f"- Human Intervention (HITL): {hitl_triggered}"
                )
            ),
        ]

        if correction_summary:
            messages.append(AIMessage(content=f"LLM Refinement Path:\n{correction_summary}"))

        # LangMem reflects on this trajectory and saves insights to the PostgresStore
        await self.memory_manager.ainvoke(
            {
                "messages": messages,
                "max_steps": 1,
            }
        )

    # ==========================================================
    # EXTERNAL INTERACTION MEMORY
    # ==========================================================

    async def process_interaction(self, messages: List[AnyMessage]) -> None:
        """Extracts patterns from direct conversations with users/agents."""
        if not self.enabled or not self.memory_manager:
            return

        await self.memory_manager.ainvoke(
            {
                "messages": messages,
                "max_steps": 1,
            }
        )

    # ==========================================================
    # RETRIEVAL (The 'Recall' Mechanism)
    # ==========================================================

    def search(self, query: str):
        """
        Queries the persistent semantic memory.
        Uses pgvector for similarity search on extracted insights.
        """
        if not self.enabled or not self.store:
            return []

        return self.store.search(
            ("thoth", "semantic_memory"),
            query=query,
        )

    # ==========================================================
    # UTILITIES
    # ==========================================================

    def is_enabled(self) -> bool:
        """Return whether semantic memory is active."""
        return self.enabled

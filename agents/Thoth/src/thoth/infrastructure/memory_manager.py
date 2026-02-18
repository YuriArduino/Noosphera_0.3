"""
Background memory management for Thoth Agent.

Uses LangMem to automatically extract long-term semantic memories
from agent interactions and internal decision trajectories.

This layer is:
- Passive (does not influence decisions directly)
- Background-driven
- Cognitively oriented (pattern consolidation)
"""

from __future__ import annotations

from typing import List, Dict, Optional

from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

from thoth.config import memory_settings


class ThothMemoryManager:
    """
    Extracts semantic knowledge from Thoth internal decisions
    and interactions using LangMem.

    Responsibilities:
        • Consolidate semantic patterns from decisions
        • Extract implicit behavioral knowledge
        • Maintain long-term vector memory
        • Operate fully in background (passive mode)
    """

    def __init__(self) -> None:
        self.enabled: bool = memory_settings.MEMORY_ENABLED

        if not self.enabled:
            self.store = None
            self.memory_manager = None
            return

        # -----------------------------------------------------
        # Shared Vector Store (Semantic Memory Layer)
        # -----------------------------------------------------
        self.store = InMemoryStore(
            index={
                "dims": memory_settings.EMBEDDING_DIMENSIONS,
                "embed": memory_settings.EMBEDDING_MODEL,
            }
        )

        # -----------------------------------------------------
        # LLM used for memory reflection extraction
        # NOTE:
        # This should ideally be a reasoning model, not embedding model.
        # Adjust via config if desired.
        # -----------------------------------------------------
        self.llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

        # -----------------------------------------------------
        # Background Memory Manager
        # -----------------------------------------------------
        self.memory_manager = create_memory_store_manager(
            self.llm,
            namespace=("thoth", "background_memory"),
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
        Extract semantic patterns from an internal decision trajectory.

        This method should be called only when:
            • Confidence is low
            • HITL is triggered
            • A correction is applied
        """
        if not self.enabled or not self.memory_manager:
            return

        messages: List[AnyMessage] = [
            SystemMessage(content="Thoth OCR agent internal reasoning log."),
            HumanMessage(
                content=(
                    f"Document ID: {document_id}\n"
                    f"Document Hash: {document_hash}\n"
                    f"Average Confidence: {avg_confidence}"
                )
            ),
            AIMessage(
                content=(
                    f"Decision Taken:\n- Action: {action}\n- Strategy: {strategy}\n"
                    f"- Attempts: {attempts}\n- HITL Triggered: {hitl_triggered}"
                )
            ),
        ]

        if correction_summary:
            messages.append(AIMessage(content=f"Correction Applied:\n{correction_summary}"))

        await self.memory_manager.ainvoke(
            {
                "messages": messages,
                "max_steps": 1,
            }
        )

    # ==========================================================
    # EXTERNAL INTERACTION MEMORY
    # ==========================================================

    async def process_interaction(
        self,
        messages: List[AnyMessage],
    ) -> None:
        """
        Extract semantic memory from conversational interactions.

        This is optional and should be used only if Thoth
        directly interacts with users.
        """
        if not self.enabled or not self.memory_manager:
            return

        await self.memory_manager.ainvoke(
            {
                "messages": messages,
                "max_steps": 1,
            }
        )

    # ==========================================================
    # SEARCH
    # ==========================================================

    def search(self, query: str):
        """
        Search background semantic memory.

        Returns:
            List of semantic memory items.
        """
        if not self.enabled or not self.store:
            return []

        return self.store.search(
            ("thoth", "background_memory"),
            query=query,
        )

    # ==========================================================
    # UTILITIES
    # ==========================================================

    def is_enabled(self) -> bool:
        """Return whether semantic memory is active."""
        return self.enabled

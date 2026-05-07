"""
Nisaba Agent Graph — LangGraph Orchestration.
Stage 2: Hybrid Memory (PostgreSQL checkpoints + pgvector semantic retrieval).
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from nisaba.config.memory import memory_settings
from nisaba.agent.node import (
    ConversationState,
    input_node,
    memory_retrieval_node,
    conversation_node,
    memory_writer_node,
)

# =============================================================================
# STATE DEFINITION
# =============================================================================


class AgentState(TypedDict):
    """Extended state for memory-aware conversation."""

    messages: List[BaseMessage]
    user_input: str
    response: str
    session_id: Optional[str]
    memory_context: Optional[str]
    should_write_memory: bool


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def build_conversation_graph():
    """
    Builds the conversation graph with optional memory nodes.
    Checkpointer is injected at compile time via factory.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("input", input_node)
    workflow.add_node("memory_retrieve", memory_retrieval_node)
    workflow.add_node("conversation", conversation_node)
    workflow.add_node("memory_write", memory_writer_node)

    # Define flow: Input → [Memory Retrieve] → Conversation → [Memory Write] → END
    workflow.set_entry_point("input")
    workflow.add_edge("input", "memory_retrieve")

    # Conditional: skip memory retrieval for short queries or if disabled
    def route_after_retrieve(state: AgentState) -> str:
        if not memory_settings.MEMORY_ENABLED:
            return "conversation"
        if len(state.get("user_input", "").strip()) < 10:
            return "conversation"
        return "conversation"  # Always proceed, memory_context is optional

    workflow.add_conditional_edges("memory_retrieve", route_after_retrieve)
    workflow.add_edge("conversation", "memory_write")
    workflow.add_edge("memory_write", END)

    # Note: checkpointer is injected at compile() time by factory
    return workflow

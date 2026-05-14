"""
Nisaba Agent Graph — LangGraph Orchestration.
Stage 2: Hybrid Memory (PostgreSQL checkpoints + pgvector semantic retrieval).
"""

from langgraph.graph import StateGraph, END
from nisaba.agent.node import (
    ConversationState,
    input_node,
    memory_retrieval_node,
    conversation_node,
    memory_writer_node,
)


def build_conversation_graph():
    """Construct the conversation graph with defined nodes and edges."""
    workflow = StateGraph(ConversationState)

    workflow.add_node("input", input_node)
    workflow.add_node("memory_retrieve", memory_retrieval_node)
    workflow.add_node("conversation", conversation_node)
    workflow.add_node("memory_write", memory_writer_node)

    workflow.set_entry_point("input")
    workflow.add_edge("input", "memory_retrieve")
    workflow.add_edge("memory_retrieve", "conversation")
    workflow.add_edge("conversation", "memory_write")
    workflow.add_edge("memory_write", END)

    return workflow

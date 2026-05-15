"""
Graph Orchestration — Nisaba Agent.

Defines the LangGraph state machine, nodes, and edges for the hybrid
memory pipeline (PostgreSQL Checkpoints + pgvector Retrieval).
"""

from langgraph.graph import StateGraph, END
from nisaba.agent.node import (
    ConversationState,
    input_node,
    memory_retrieval_node,
    knowledge_retrieval_node,
    conversation_node,
    memory_writer_node,
    entity_extraction_node,
)


def build_conversation_graph() -> StateGraph:
    """
    Constructs the LangGraph StateGraph for Nisaba's cognitive cycle.

    The graph follows a sequential 'Scan-Think-Commit' architecture:
    1. Input: Sanitizes user request.
    2. Memory Retrieve: Fetches context from pgvector (SST).
    3. Knowledge Retrieve: Fetches relational context from Neo4j when enabled.
    4. Conversation: LLM reasoning with retrieved context.
    5. Memory Write: Commits the turn to the long-term vector store.

    Returns:
        An uncompiled StateGraph instance.

    Design rationale:
        - Decoupling the build from the compilation allows the factory to
          inject different checkpointers (Postgres vs Memory) dynamically.
        - Using a structured ConversationState ensures that internal logic
          remains type-safe across node transitions.

    Use cases:
        - Standard chat interaction with psychoanalytic context retrieval.
        - Automated reflection cycles where memory_write triggers insights.
    """
    # Initialize the state machine with our specialized ConversationState
    workflow = StateGraph(ConversationState)

    # ---------------------------------------------------------------------------
    # NODE REGISTRATION
    # ---------------------------------------------------------------------------
    workflow.add_node("input", input_node)
    workflow.add_node("memory_retrieve", memory_retrieval_node)
    workflow.add_node("knowledge_retrieve", knowledge_retrieval_node)
    workflow.add_node("conversation", conversation_node)
    workflow.add_node("memory_write", memory_writer_node)
    workflow.add_node("entity_extract", entity_extraction_node)

    # ---------------------------------------------------------------------------
    # EDGE DEFINITION (COGNITIVE PIPELINE)
    # ---------------------------------------------------------------------------
    workflow.set_entry_point("input")

    # Linear flow for Stage 2 (Hybrid Memory)
    workflow.add_edge("input", "memory_retrieve")
    workflow.add_edge("memory_retrieve", "knowledge_retrieve")
    workflow.add_edge("knowledge_retrieve", "conversation")
    workflow.add_edge("conversation", "memory_write")

    # Persist extracted relational facts after the response is safely generated.
    workflow.add_edge("memory_write", "entity_extract")
    workflow.add_edge("entity_extract", END)

    return workflow

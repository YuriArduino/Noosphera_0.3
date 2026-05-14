"""
Graph Factory — Noosphera Nisaba.

Manages the lifecycle of conversation graphs, providing a context manager
to handle persistent state via PostgreSQL or ephemeral MemorySaver.
"""

from contextlib import contextmanager
from typing import Generator, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# Internal imports from Nisaba's specific config and the shared infrastructure
from agents.shared.config.memory import memory_settings
from nisaba.agent.graph import build_conversation_graph


@contextmanager
def get_conversation_graph(session_id: Optional[str] = None) -> Generator:
    """
    Context manager that yields a compiled graph with the appropriate checkpointer.

    Args:
        session_id: Optional identifier used as thread_id for state persistence.

    Yields:
        Compiled LangGraph instance with an active checkpointer.

    Use cases:
        - Production: Persistent storage in PostgreSQL for Human-in-the-Loop (HITL).
        - Development/Test: Fast in-memory state tracking.

    Design rationale:
        - Using a context manager ensures that database connections are properly
          pooled and closed after the graph execution, preventing memory leaks.
        - PostgresSaver.setup() is called to ensure tables exist in the SST database.
    """

    # ---------------------------------------------------------------------------
    # DEVELOPMENT MODE: In-memory checkpointer
    # ---------------------------------------------------------------------------
    if not memory_settings.CHECKPOINT_ENABLED:
        checkpointer = MemorySaver()
        graph_factory = build_conversation_graph()

        # We yield the compiled graph directly for ephemeral sessions
        yield graph_factory.compile(checkpointer=checkpointer)
        return

    # ---------------------------------------------------------------------------
    # PRODUCTION MODE: PostgreSQL persistence (Single Source of Truth)
    # ---------------------------------------------------------------------------
    # We use the centralized DATABASE_URL from the shared memory settings.
    # Note: If running multiple agents, they share the same DB but use
    # thread_ids/namespaces to avoid state collision.
    with PostgresSaver.from_conn_string(memory_settings.DATABASE_URL) as checkpointer:
        # Idempotent setup: ensures langgraph tables exist in the 'noosphera' DB
        checkpointer.setup()

        graph_factory = build_conversation_graph()
        compiled_graph = graph_factory.compile(checkpointer=checkpointer)

        yield compiled_graph

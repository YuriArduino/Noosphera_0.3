"""
Graph Factory — Manages conversation graph lifecycle with proper checkpoint handling.
Pattern: Factory + Context Manager for PostgresSaver compatibility.
"""

from contextlib import contextmanager
from typing import Iterator, Optional
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver

from nisaba.config.memory import memory_settings
from nisaba.agent.graph import build_conversation_graph, AgentState


@contextmanager
def get_conversation_graph(session_id: Optional[str] = None):
    """
    Context manager that yields a compiled graph with appropriate checkpointer.

    Usage:
        with get_conversation_graph(session_id="abc123") as graph:
            result = graph.invoke(
                {"messages": [], "user_input": "Olá", "session_id": session_id},
                config={"configurable": {"thread_id": session_id}}
            )

    Args:
        session_id: Optional thread identifier for checkpoint retrieval

    Yields:
        Compiled LangGraph instance
    """
    # Development mode: use in-memory checkpointer for simplicity
    if not memory_settings.CHECKPOINT_ENABLED:
        checkpointer = MemorySaver()
        graph = build_conversation_graph()
        yield graph.compile(checkpointer=checkpointer)
        return

    # Production mode: PostgreSQL persistence
    with PostgresSaver.from_conn_string(
        memory_settings.NISABA_DATABASE_URL,
        # Opcional: configurações de pool para produção
        # pool_size=5, max_overflow=10
    ) as checkpointer:
        # Setup é idempotente: cria tabelas se não existirem
        checkpointer.setup()

        graph = build_conversation_graph()
        compiled = graph.compile(checkpointer=checkpointer)
        yield compiled

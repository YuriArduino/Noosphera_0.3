"""
Graph Factory — Manages conversation graph lifecycle with proper checkpoint handling.
Pattern: Factory + Context Manager for PostgresSaver compatibility.
"""

from contextlib import contextmanager
from typing import Iterator, Optional
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver

from nisaba.config.memory import memory_settings
from nisaba.agent.graph import build_conversation_graph


@contextmanager
def get_conversation_graph(session_id: Optional[str] = None):
    """
    Context manager that yields a compiled graph with appropriate checkpointer.

    Usage:
        with get_conversation_graph() as graph:
            result = graph.invoke(
                {"messages": [], "user_input": "Olá", "session_id": session_id},
                config={"configurable": {"thread_id": session_id}}
            )

    Args:
        session_id: Optional identifier, used later as thread_id for checkpointing.
                    The factory itself does not depend on it; it's passed for consistency.

    Yields:
        Compiled LangGraph instance with active checkpointer.
    """
    # Development mode: in-memory checkpointer
    if not memory_settings.CHECKPOINT_ENABLED:
        checkpointer = MemorySaver()
        graph = build_conversation_graph()
        yield graph.compile(checkpointer=checkpointer)
        return

    # Production mode: PostgreSQL persistence
    with PostgresSaver.from_conn_string(
        memory_settings.NISABA_DATABASE_URL,
        # Ajuste de pool opcional para produção:
        # pool_size=5, max_overflow=10
    ) as checkpointer:
        checkpointer.setup()  # idempotente: cria tabelas se não existirem
        graph = build_conversation_graph()
        compiled = graph.compile(checkpointer=checkpointer)
        yield compiled

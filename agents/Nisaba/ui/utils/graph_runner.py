"""
Graph Runner — Safe invocation of LangGraph with proper context management.
"""

from typing import Any, Dict
from contextlib import contextmanager
import streamlit as st


@contextmanager
def get_conversation_graph(session_id: str):
    """
    Context manager for conversation graph with proper checkpoint lifecycle.
    Falls back to MemorySaver if PostgreSQL is unavailable.
    """
    try:
        from nisaba.agent.factory import get_conversation_graph as factory_get_graph

        with factory_get_graph(session_id=session_id) as graph:
            yield graph
    except ImportError:
        # Fallback for development: use graph without persistent checkpointer
        from nisaba.agent.graph import conversation_graph

        yield conversation_graph
    except Exception as e:
        st.warning(f"⚠️ Fallback para memória em memória: {e}")
        from langgraph.checkpoint.memory import MemorySaver
        from nisaba.agent.graph import build_conversation_graph

        graph = build_conversation_graph().compile(checkpointer=MemorySaver())
        yield graph


def invoke_conversation_graph(user_input: str, session_id: str) -> Dict[str, Any]:
    """
    Invoke the conversation graph with proper error handling.

    Args:
        user_input: The user's message
        session_id: Unique identifier for the conversation thread

    Returns:
        Dict with 'response', 'messages', and optional 'memory_context'
    """
    with get_conversation_graph(session_id) as graph:
        result = graph.invoke(
            {
                "messages": [],  # Will be populated by graph state
                "user_input": user_input,
                "session_id": session_id,
                "response": "",
            },
            config={"configurable": {"thread_id": session_id}},
        )

    return {
        "response": result.get("response", ""),
        "messages": result.get("messages", []),
        "memory_context": result.get("memory_context"),
    }

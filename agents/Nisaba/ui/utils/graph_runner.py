"""
Graph Runner — Safe invocation of LangGraph with proper context management.
"""

import logging
from typing import Any, Dict, Optional, List
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


def invoke_conversation_graph(
    user_input: str,
    session_id: str,
    messages: Optional[List[BaseMessage]] = None,
) -> Dict[str, Any]:
    """Invoke the conversation graph with proper error handling and context management."""
    try:
        # Nome real do seu arquivo: graph_factory.py
        from nisaba.agent.graph_factory import get_conversation_graph

        with get_conversation_graph() as graph:
            state = {
                "messages": messages or [],
                "user_input": user_input,
                "session_id": session_id,
                "response": "",
                "memory_context": None,
                "should_write_memory": False,
            }
            config = {"configurable": {"thread_id": session_id}}
            result = graph.invoke(state, config)

            return {
                "response": result.get("response", ""),
                "messages": result.get("messages", []),
                "memory_context": result.get("memory_context"),
            }
    except ImportError:
        # Fallback – só será usado se graph_factory.py não existir
        logger.warning("Factory unavailable, using MemorySaver fallback")
        from nisaba.agent.graph import build_conversation_graph
        from langgraph.checkpoint.memory import MemorySaver

        graph = build_conversation_graph().compile(checkpointer=MemorySaver())
        state = {
            "messages": messages or [],
            "user_input": user_input,
            "session_id": session_id,
            "response": "",
            "memory_context": None,
            "should_write_memory": False,
        }
        result = graph.invoke(state, {"configurable": {"thread_id": session_id}})
        return {
            "response": result.get("response", ""),
            "messages": result.get("messages", []),
            "memory_context": result.get("memory_context"),
        }
    except Exception as e:
        logger.error("Graph invocation failed: %s", e)
        return {
            "response": f"⚠️ Erro no grafo: {type(e).__name__}",
            "messages": messages or [],
            "memory_context": None,
        }

"""
Debug Panel — Developer tools for troubleshooting (dev-only).
"""

import streamlit as st
from typing import Any, Dict


def render_debug_panel(result: Dict[str, Any], state_mgr: Any) -> None:
    """
    Render debug information when debug mode is enabled.

    Args:
        result: Graph invocation result
        state_mgr: SessionStateManager instance
    """
    if not st.session_state.get("debug_mode", False):
        return

    with st.expander("🔧 Painel de Debug", expanded=False):
        st.subheader("📦 Estado da Sessão")
        st.json(
            {
                "session_id": state_mgr.session_id,
                "message_count": len(state_mgr.messages),
                "graph_message_count": len(state_mgr.graph_messages),
            },
            expanded=False,
        )

        if result.get("memory_context"):
            st.subheader("🧠 Contexto de Memória")
            st.markdown(f"```{result['memory_context']}```")

        st.subheader("🔗 Configuração Ativa")
        try:
            from nisaba.config.cognition import memory_settings

            st.json(
                {
                    "MEMORY_ENABLED": memory_settings.MEMORY_ENABLED,
                    "CHECKPOINT_ENABLED": memory_settings.CHECKPOINT_ENABLED,
                    "VECTORSTORE_ENABLED": memory_settings.VECTORSTORE_ENABLED,
                    "SEMANTIC_SEARCH_TOP_K": memory_settings.SEMANTIC_SEARCH_TOP_K,
                },
                expanded=False,
            )
        except Exception as e:
            st.error(f"Erro ao carregar config: {e}")

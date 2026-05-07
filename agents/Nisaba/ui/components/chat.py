"""
Chat Interface Components — Render chat UI with proper error handling.
"""

import streamlit as st
from typing import Callable, Any
from ui.state.manager import SessionStateManager


def render_chat_interface(
    state_mgr: SessionStateManager, graph_invoker: Callable[[str, str], Any]
) -> None:
    """
    Render the main chat interface.

    Args:
        state_mgr: Session state manager instance
        graph_invoker: Function that invokes the LangGraph (user_input, session_id) -> result
    """
    # Display message history
    for msg in state_mgr.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input handler
    if user_input := st.chat_input("Digite sua mensagem...", key="chat_input"):
        _handle_user_input(state_mgr, graph_invoker, user_input)


def _handle_user_input(
    state_mgr: SessionStateManager, graph_invoker: Callable[[str, str], Any], user_input: str
) -> None:
    """Process user input and invoke the conversation graph."""
    # Add user message to UI
    state_mgr.add_user_message(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"), st.spinner("🤔 Pensando..."):
        try:
            result = graph_invoker(user_input, state_mgr.session_id)
            response = result.get("response", "⚠️ Não foi possível gerar uma resposta.")

            # Update state and display
            state_mgr.add_assistant_message(response)
            st.markdown(response)

            # Show memory context if available (debug feature)
            if result.get("memory_context"):
                with st.expander("🧠 Contexto de Memória", expanded=False):
                    st.markdown(f"```{result['memory_context']}```")

        except Exception as e:
            error_msg = f"Erro: `{type(e).__name__}`"
            state_mgr.set_error(error_msg)
            st.error(f"❌ {error_msg}")
            with st.expander("🔍 Detalhes do erro", expanded=False):
                st.code(str(e), language="text")

"""
Sidebar Component — Configuration panel and controls.
"""

import streamlit as st
from ui.state.manager import SessionStateManager


def render_sidebar(state_mgr: SessionStateManager) -> None:
    """Render the sidebar with configuration and controls."""
    with st.sidebar:
        st.header("⚙️ Configuração")

        # LLM Configuration
        _render_llm_config()

        st.divider()

        # Memory Status
        _render_memory_status(state_mgr)

        st.divider()

        # Controls
        _render_controls(state_mgr)

        st.divider()
        st.caption("🔖 Nisaba v0.3.0")


def _render_llm_config() -> None:
    """Display LLM connection settings."""
    try:
        from nisaba.config.llm import llm_settings

        st.info(f"""
        **Conexão**
        - Base URL: `{llm_settings.LLM_BASE_URL}`
        - API Key: `{'••••' if llm_settings.LLM_API_KEY else 'None'}`

        **Modelo**
        - Nome: `{llm_settings.CHAT_MODEL}`
        - Temperatura: `{llm_settings.CHAT_TEMPERATURE}`
        - Max Tokens: `{llm_settings.CHAT_MAX_TOKENS}`

        **Embeddings**
        - Modelo: `{llm_settings.EMBEDDING_MODEL}`
        - Dimensão: `{llm_settings.EMBEDDING_DIMENSION}`
        """)
    except Exception as e:
        st.warning(f"⚠️ Não foi possível carregar configurações do LLM: {e}")


def _render_memory_status(state_mgr: SessionStateManager) -> None:
    """Display memory system status."""
    status = "🟢 Ativa" if state_mgr.memory_enabled else "🔴 Inativa"
    st.success(f"**Memória**: {status}")

    if state_mgr.memory_enabled:
        try:
            from nisaba.config.memory import memory_settings

            st.caption(f"""
            - Checkpoints: {'✅' if memory_settings.CHECKPOINT_ENABLED else '❌'}
            - Semântica: {'✅' if memory_settings.VECTORSTORE_ENABLED else '❌'}
            - Top-K: {memory_settings.SEMANTIC_SEARCH_TOP_K}
            """)
        except:
            pass


def _render_controls(state_mgr: SessionStateManager) -> None:
    """Render action buttons."""
    st.subheader("🎛️ Controles")

    if st.button("🗑️ Limpar conversa", type="secondary", use_container_width=True):
        state_mgr.clear()
        st.rerun()

    if st.button("🔄 Nova sessão", type="secondary", use_container_width=True):
        state_mgr.new_session()
        st.rerun()

    # Debug toggle (dev-only)
    if st.checkbox("🔧 Modo debug", key="debug_mode", value=False):
        st.caption("Debug ativado: mostre detalhes de memória e erros")

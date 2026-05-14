"""
Sidebar Component — Configuration panel and controls.
"""

import streamlit as st
from ui.state.manager import SessionStateManager


def render_sidebar(state_mgr: SessionStateManager) -> None:
    with st.sidebar:
        st.header("⚙️ Configuração")
        _render_llm_config()
        st.divider()
        _render_memory_status(state_mgr)
        st.divider()
        _render_controls(state_mgr)
        st.divider()
        st.caption("🔖 Nisaba v0.3.0")


def _render_controls(state_mgr: SessionStateManager) -> None:
    """Render action buttons."""
    st.subheader("🎛️ Controles")

    if st.button("🗑️ Limpar conversa", type="secondary", use_container_width=True):
        state_mgr.clear()
        st.rerun()

    if st.button("🔄 Nova sessão", type="secondary", use_container_width=True):
        state_mgr.new_session()
        st.rerun()

    # Toggle para visualização do grafo
    if "show_graph" not in st.session_state:
        st.session_state.show_graph = False

    if st.button("🔍 Visualizar Grafo", type="secondary", use_container_width=True):
        st.session_state.show_graph = not st.session_state.show_graph
        st.rerun()

    if st.session_state.show_graph:
        _render_graph_visualization()

    # Debug toggle
    if st.checkbox("🔧 Modo debug", key="debug_mode", value=False):
        st.caption("Debug ativado: mostre detalhes de memória e erros")


def _render_graph_visualization():
    """Gera e exibe o grafo de conversação usando Mermaid."""
    try:
        from nisaba.agent.graph import build_conversation_graph
        from langgraph.checkpoint.memory import MemorySaver

        # Construímos e compilamos com um MemorySaver temporário
        graph_builder = build_conversation_graph()
        compiled_graph = graph_builder.compile(checkpointer=MemorySaver())

        # Obtém a definição Mermaid
        mermaid_code = compiled_graph.get_graph().draw_mermaid()

        # HTML que carrega a biblioteca Mermaid e renderiza o gráfico
        mermaid_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{ startOnLoad: true }});</script>
        </head>
        <body>
            <div class="mermaid">
                {mermaid_code}
            </div>
        </body>
        </html>
        """
        st.components.v1.html(mermaid_html, height=400, scrolling=True)

        with st.expander("📝 Código Mermaid"):
            st.code(mermaid_code, language="mermaid")

    except Exception as e:
        st.error(f"Erro ao gerar grafo: {e}")


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
            from nisaba.config.cognition import memory_settings

            st.caption(f"""
            - Checkpoints: {'✅' if memory_settings.CHECKPOINT_ENABLED else '❌'}
            - Semântica: {'✅' if memory_settings.VECTORSTORE_ENABLED else '❌'}
            - Top-K: {memory_settings.SEMANTIC_SEARCH_TOP_K}
            """)
        except:
            pass

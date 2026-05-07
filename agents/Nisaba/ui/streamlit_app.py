#!/usr/bin/env python3
"""
Streamlit Entry Point — Nisaba Agent.
Minimal bootstrap that delegates to modular components.
"""

import streamlit as st
from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat_interface
from ui.state.manager import SessionStateManager
from ui.utils.graph_runner import invoke_conversation_graph

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="🧠 Nisaba Agent",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={"About": "Nisaba Agent v0.3.0 — Memória híbrida com PostgreSQL + pgvector"},
)

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================
state_mgr = SessionStateManager()
state_mgr.initialize()

# =============================================================================
# SIDEBAR
# =============================================================================
render_sidebar(state_mgr)

# =============================================================================
# MAIN INTERFACE
# =============================================================================
st.title("🧠 Nisaba Agent")
st.caption(
    f"Sessão: `{state_mgr.session_id}` • Memória: {'🟢 Ativa' if state_mgr.memory_enabled else '🔴 Inativa'}"
)

# Chat interface
render_chat_interface(state_mgr, invoke_conversation_graph)

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption(
    "🔖 Nisaba v0.3.0 • "
    "PostgreSQL Checkpoints + pgvector Semantic Memory • "
    "Preparado para Neo4j Knowledge Graph"
)

#!/usr/bin/env python3
"""
Streamlit Entry Point — Nisaba Agent.
Minimal bootstrap that delegates to modular components.
"""

import sys
import os
from pathlib import Path

# =============================================================================
# PATH RESOLUTION — Critical for Streamlit + package imports
# =============================================================================
# __file__ = /agents/Nisaba/ui/streamlit_app.py
# We need to add /agents/Nisaba/src to sys.path for `from nisaba.*` imports
# AND /agents/Nisaba to sys.path for `from ui.*` imports

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent  # /agents/Nisaba
REPO_ROOT = PROJECT_ROOT.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Add to sys.path if not already present
for path in [str(REPO_ROOT), str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Load .env before any config imports
from dotenv import load_dotenv

env_path = REPO_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)

# =============================================================================
# NOW IMPORTS WORK
# =============================================================================
import streamlit as st
from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat_interface
from ui.state.manager import SessionStateManager
from ui.utils.graph_runner import invoke_conversation_graph
from nisaba.config.cognition import memory_settings

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
memory_status = "🟢 Ativa" if memory_settings.MEMORY_ENABLED else "🔴 Inativa"
st.caption(f"Sessão: `{state_mgr.session_id}` • Memória: {memory_status}")

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

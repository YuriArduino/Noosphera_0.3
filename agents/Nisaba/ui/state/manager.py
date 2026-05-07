"""
Session State Manager — Type-safe wrapper for Streamlit session_state.
Centralizes all state variables with clear typing and lifecycle methods.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


@dataclass
class ChatState:
    """Typed container for chat-related state."""

    messages: List[Dict[str, str]] = field(default_factory=list)
    graph_messages: List[BaseMessage] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    last_response: Optional[str] = None
    error: Optional[str] = None


class SessionStateManager:
    """Manages Streamlit session state with type safety and convenience methods."""

    def __init__(self):
        self._state = st.session_state

    def initialize(self) -> None:
        """Initialize all state variables if not present."""
        if "_chat" not in self._state:
            self._state._chat = ChatState()

    @property
    def chat(self) -> ChatState:
        """Typed access to chat state."""
        return self._state._chat

    @property
    def session_id(self) -> str:
        return self.chat.session_id

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self.chat.messages

    @property
    def graph_messages(self) -> List[BaseMessage]:
        return self.chat.graph_messages

    @property
    def memory_enabled(self) -> bool:
        """Check if memory features are enabled (from env/config)."""
        try:
            from nisaba.config.memory import memory_settings

            return memory_settings.MEMORY_ENABLED
        except:
            return False

    def add_user_message(self, content: str) -> None:
        """Add a user message to both visual and graph histories."""
        self.messages.append({"role": "user", "content": content})
        self.graph_messages.append(HumanMessage(content=content))

    def add_assistant_message(self, content: str, graph_msg: Optional[BaseMessage] = None) -> None:
        """Add an assistant response to histories."""
        self.messages.append({"role": "assistant", "content": content})
        if graph_msg:
            self.graph_messages.append(graph_msg)
        self.chat.last_response = content

    def set_error(self, error: str) -> None:
        """Record an error for display."""
        self.chat.error = error
        self.add_assistant_message(f"⚠️ {error}")

    def clear(self) -> None:
        """Reset conversation state (keeps session_id)."""
        self.chat.messages = []
        self.chat.graph_messages = []
        self.chat.last_response = None
        self.chat.error = None

    def new_session(self) -> None:
        """Start a fresh session with new ID."""
        self.chat.session_id = str(uuid.uuid4())[:8]
        self.clear()

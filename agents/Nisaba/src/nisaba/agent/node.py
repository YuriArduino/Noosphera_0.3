"""
Nisaba Agent Nodes — LangGraph Execution Units with Memory.
Stage 2: Hybrid Memory Integration.
"""

from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Config imports
from nisaba.config.llm import llm_settings
from nisaba.config.memory import memory_settings

# Memory imports (SSOT)
from nisaba.schema.tables import SemanticExperienceTable
from nisaba.memory.vector_store import VectorStore

# =============================================================================
# STATE DEFINITION
# =============================================================================


class ConversationState(TypedDict):
    """Core conversation state for LangGraph."""

    messages: List[BaseMessage]
    user_input: str
    response: str
    session_id: Optional[str]
    memory_context: Optional[str]
    should_write_memory: bool


# =============================================================================
# LLM INITIALIZATION
# =============================================================================


def initialize_llm():
    """Initialize LLM with configured settings."""
    return ChatOpenAI(
        model=llm_settings.CHAT_MODEL,
        openai_api_key=llm_settings.LLM_API_KEY,
        openai_api_base=llm_settings.LLM_BASE_URL,
        temperature=llm_settings.CHAT_TEMPERATURE,
        max_tokens=llm_settings.CHAT_MAX_TOKENS,
        timeout=llm_settings.LLM_TIMEOUT,
        max_retries=llm_settings.LLM_MAX_RETRIES,
    )


# =============================================================================
# CORE NODES
# =============================================================================


def input_node(state: ConversationState) -> ConversationState:
    """Validate and prepare user input."""
    user_input = state.get("user_input", "").strip()
    if not user_input:
        return {"response": "Por favor, envie uma mensagem válida."}

    # Ensure session_id exists
    if not state.get("session_id"):
        import uuid

        state["session_id"] = str(uuid.uuid4())[:8]

    return {"user_input": user_input, "session_id": state["session_id"]}


def memory_retrieval_node(state: ConversationState) -> ConversationState:
    """
    Retrieve similar past experiences to inform the response.
    Uses pgvector similarity search with fallback safety.
    """
    if not memory_settings.MEMORY_ENABLED or not memory_settings.VECTORSTORE_ENABLED:
        return state

    query = state.get("user_input", "")
    if len(query.strip()) < 10:  # Skip very short queries
        return state

    try:
        vector_store = VectorStore()
        similar = vector_store.search_similar(
            query=query, limit=memory_settings.SEMANTIC_SEARCH_TOP_K, min_relevance=0.3
        )

        if similar:
            context_snippets = [
                f"[Experiência #{i+1}] {exp.title or 'Sem título'}: {exp.content[:200]}..."
                for i, exp in enumerate(similar[:3])
            ]
            state["memory_context"] = "\n\n".join(context_snippets)

            # Track usage for relevance ranking
            for exp in similar[:1]:  # Only top result
                vector_store.increment_usage(exp.id)

    except Exception as e:
        # Fallback: log error but don't break the conversation
        print(f"⚠️ Memory retrieval failed: {e}")
        state["memory_context"] = None

    return state


def conversation_node(state: ConversationState) -> ConversationState:
    """
    Main conversation node: generates response using LLM.
    Injects memory context if available.
    """
    llm = initialize_llm()
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")

    # Build prompt with optional memory context
    system_prompt = "Você é Nisaba, um assistente conversacional com memória persistente."

    memory_context = state.get("memory_context")
    if memory_context:
        system_prompt += f"\n\n🧠 Contexto de experiências similares:\n{memory_context}"

    # Prepare messages for LLM
    llm_messages = (
        [SystemMessage(content=system_prompt)] + messages + [HumanMessage(content=user_input)]
    )

    try:
        response = llm.invoke(llm_messages)
        state["messages"] = llm_messages + [response]
        state["response"] = response.content
        state["should_write_memory"] = True  # Flag for writer node
    except Exception as e:
        state["response"] = f"⚠️ Erro ao processar: {type(e).__name__}"
        state["should_write_memory"] = False

    return state


def memory_writer_node(state: ConversationState) -> ConversationState:
    """
    Persist interaction to semantic memory for future retrieval.
    Runs asynchronously to avoid blocking the conversation.
    """
    if not state.get("should_write_memory") or not memory_settings.MEMORY_ENABLED:
        return state

    try:
        vector_store = VectorStore()

        # Extract content for embedding
        user_input = state.get("user_input", "")
        agent_response = state.get("response", "")
        content = f"Usuário: {user_input}\nNisaba: {agent_response}"

        # Add to semantic memory
        vector_store.add_experience(
            content=content,
            session_id=state.get("session_id", "unknown"),
            title=f"Interação: {user_input[:50]}...",
            category="conversation",
            tags=["user_query", "agent_response"],
            metadata_={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tokens_estimated": len(content.split()) * 1.3,
            },
        )

    except Exception as e:
        # Fallback: log but don't break
        print(f"⚠️ Memory write failed: {e}")

    return state

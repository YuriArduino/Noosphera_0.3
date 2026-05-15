"""
Nisaba Agent Nodes — Noosphera Execution Units.

Implements the functional logic for each step of the LangGraph cycle,
integrating hybrid memory retrieval and persistence via centralized prompts.
"""

import logging
import re
import uuid
from typing import Any, List, Optional, TypedDict
from datetime import datetime, timezone
from functools import lru_cache

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase

# Configuration & Infrastructure
from agents.shared.config.memory import memory_settings
from nisaba.config.llm import nisaba_llm_settings
from nisaba.memory.vector_store import VectorStore
from nisaba.agent.prompt import NisabaPrompts

logger = logging.getLogger(__name__)

# =============================================================================
# STATE DEFINITION
# =============================================================================


class ConversationState(TypedDict):
    """Schema for the Nisaba cognitive state within the LangGraph lifecycle."""

    messages: List[BaseMessage]
    user_input: str
    response: str
    session_id: Optional[str]
    memory_context: Optional[str]
    should_write_memory: bool


# =============================================================================
# INFRASTRUCTURE SINGLETONS
# =============================================================================


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Lazy initialization of the VectorStore connection (pgvector)."""
    return VectorStore()


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Provides a cached instance of the LLM using Nisaba-specific settings."""
    return ChatOpenAI(
        model=nisaba_llm_settings.CHAT_MODEL,
        api_key=nisaba_llm_settings.LLM_API_KEY,
        base_url=nisaba_llm_settings.LLM_BASE_URL,
        temperature=nisaba_llm_settings.CHAT_TEMPERATURE,
        max_tokens=nisaba_llm_settings.CHAT_MAX_TOKENS,
        timeout=nisaba_llm_settings.LLM_TIMEOUT,
        max_retries=nisaba_llm_settings.LLM_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def get_neo4j_driver():
    """Lazy initialization of the Neo4j driver used by knowledge retrieval."""
    return GraphDatabase.driver(
        memory_settings.NEO4J_URI,
        auth=(memory_settings.NEO4J_USER, memory_settings.NEO4J_PASSWORD),
    )


# =============================================================================
# UTILITIES
# =============================================================================


def clean_reasoning(text: str) -> str:
    """Removes internal <thought> or <think> tags for final user delivery."""
    return re.sub(r"<(thought|think)>.*?</\1>", "", text, flags=re.DOTALL).strip()


def _clean_cypher(text: str) -> str:
    """Extracts a single Cypher statement from an LLM response."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:cypher)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    return cleaned.rstrip(";").strip()


def _is_safe_readonly_cypher(cypher: str) -> bool:
    """Allow only read-oriented Cypher generated for retrieval."""
    normalized = re.sub(r"\s+", " ", cypher.strip()).upper()
    if not normalized.startswith(("MATCH ", "OPTIONAL MATCH ", "WITH ", "RETURN ", "CALL ")):
        return False

    forbidden = {
        "CREATE",
        "DELETE",
        "DETACH",
        "DROP",
        "FOREACH",
        "LOAD CSV",
        "MERGE",
        "REMOVE",
        "SET",
    }
    return not any(re.search(rf"\b{term}\b", normalized) for term in forbidden)


def _format_knowledge_rows(rows: List[dict[str, Any]], max_chars: int = 800) -> str:
    """Renders Neo4j rows compactly for injection into memory_context."""
    rendered = str(rows)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + "..."
    return "Knowledge Graph:\n" + rendered


# =============================================================================
# CORE NODES
# =============================================================================


def input_node(state: ConversationState) -> dict:
    """Sanitizes input and initializes session tracking."""
    user_input = state.get("user_input", "").strip()

    if not user_input:
        return {"response": "Please provide a valid input."}

    if not state.get("session_id"):
        state["session_id"] = str(uuid.uuid4())[:8]

    return {"user_input": user_input, "session_id": state["session_id"]}


def memory_retrieval_node(state: ConversationState) -> dict:
    """Retrieves relevant semantic context from the Single Source of Truth (SST)."""
    if not memory_settings.MEMORY_ENABLED or not memory_settings.VECTORSTORE_ENABLED:
        return {}

    query = state.get("user_input", "")
    if len(query.strip()) < 10:
        return {}

    try:
        vs = get_vector_store()
        similar = vs.search_hybrid(
            query=query,
            limit=memory_settings.SEMANTIC_SEARCH_TOP_K,
        )

        if similar:
            context_snippets = [
                f"[Experience #{i+1}] {exp.title or 'Untitled'}: {exp.content[:200]}..."
                for i, exp in enumerate(similar[:3])
            ]

            # Increment usage for the primary hit to support ranking over time
            if similar[0].id is not None:
                vs.increment_usage(similar[0].id)

            return {"memory_context": "\n\n".join(context_snippets)}

    except Exception as e:
        logger.warning(f"Memory retrieval failed: {e}")

    return {}


def knowledge_retrieval_node(state: ConversationState) -> dict:
    """Retrieves relational context from Neo4j using LLM-generated read-only Cypher."""
    if not memory_settings.MEMORY_ENABLED or not memory_settings.KNOWLEDGE_GRAPH_ENABLED:
        return {}

    query = state.get("user_input", "")
    if len(query.strip()) < 5:
        return {}

    llm = get_llm()
    text2cypher_prompt = f"""
You are a Neo4j Cypher specialist. Given this knowledge graph schema:
- (:Person {{name}})-[:ASKED_ABOUT {{date}}]->(:Topic {{name}})
- (:Person)-[:EXPRESSED_PREFERENCE]->(:Preference {{key, value}})
- (:Session {{id, summary, timestamp}})-[:MENTIONS]->(:Topic)
- (:Experience {{id, content}})-[:BELONGS_TO_THEME]->(:Theme {{name}})

Convert the user's question into one safe read-only Cypher query.
Use only MATCH, OPTIONAL MATCH, WITH, RETURN, ORDER BY, LIMIT, and read-only CALL clauses.
Never use CREATE, MERGE, SET, DELETE, DETACH, REMOVE, DROP, or LOAD CSV.
Return only the Cypher query, with no explanations or Markdown.

Question: {query}
""".strip()

    try:
        response = llm.invoke([HumanMessage(content=text2cypher_prompt)])
        cypher_query = _clean_cypher(str(response.content))

        if not _is_safe_readonly_cypher(cypher_query):
            logger.warning("Cypher query blocked by read-only guard: %s", cypher_query)
            return {}

        driver = get_neo4j_driver()
        with driver.session(database=None) as session:
            records = session.run(cypher_query).data()

        if not records:
            return {}

        context = _format_knowledge_rows(records)
        existing_context = state.get("memory_context")
        if existing_context:
            context = f"{existing_context}\n\n{context}"

        return {"memory_context": context}

    except Exception as e:
        logger.warning("Knowledge retrieval failed: %s", e)

    return {}


def conversation_node(state: ConversationState) -> dict:
    """Core reasoning node: invokes LLM with personality and context."""
    llm = get_llm()
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")

    # Retrieve centralized prompt from prompt.py
    system_prompt = NisabaPrompts.get_main_system_prompt(memory_context=state.get("memory_context"))

    llm_messages = (
        [SystemMessage(content=system_prompt)] + messages + [HumanMessage(content=user_input)]
    )

    try:
        response = llm.invoke(llm_messages)
        return {
            "messages": llm_messages + [response],
            "response": clean_reasoning(response.content),
            "should_write_memory": True,
        }
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        return {
            "response": f"⚠️ Error: {type(e).__name__}",
            "should_write_memory": False,
        }


def memory_writer_node(state: ConversationState) -> dict:
    """Persists the current interaction back to the vector store."""
    if not state.get("should_write_memory") or not memory_settings.VECTORSTORE_ENABLED:
        return {}

    try:
        vs = get_vector_store()
        user_input = state.get("user_input", "")
        agent_response = state.get("response", "")

        content = f"User: {user_input}\nNisaba: {agent_response}"

        vs.add_experience(
            content=content,
            session_id=state.get("session_id") or "unknown",
            title=f"Chat: {user_input[:50]}...",
            category="conversation",
            tags=["interaction", "nisaba_0.3"],
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": nisaba_llm_settings.CHAT_MODEL,
            },
        )
    except Exception as e:
        logger.warning(f"Memory write failed: {e}")

    return {}

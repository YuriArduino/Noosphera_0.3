"""
Nisaba Agent Nodes — Noosphera Execution Units.

Implements the functional logic for each step of the LangGraph cycle,
integrating hybrid memory retrieval and persistence via centralized prompts.
"""

import logging
import json
import os
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

DEFAULT_PERSON_ID = os.getenv("NISABA_PERSON_ID", "local_user")
DEFAULT_PERSON_NAME = os.getenv("NISABA_PERSON_NAME", "Yuri")

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


def _clean_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)

    parsed = json.loads(cleaned)
    return parsed if isinstance(parsed, dict) else {}


def _slug(value: str, fallback: str = "unknown") -> str:
    """Create a small stable identifier segment for graph node IDs."""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or fallback


def _normalize_preference_key(key: str) -> str:
    """Collapse common preference variants into canonical keys."""
    normalized = _slug(key)
    aliases = {
        "animal": "favorite_animal",
        "animais": "favorite_animal",
        "favorite_pet": "favorite_animal",
        "animal_favorito": "favorite_animal",
        "favorite_animal": "favorite_animal",
        "comida": "favorite_food",
        "comida_favorita": "favorite_food",
        "favorite_food": "favorite_food",
        "cor": "favorite_color",
        "cor_favorita": "favorite_color",
        "favorite_color": "favorite_color",
    }
    return aliases.get(normalized, normalized)


def _normalize_preference_value(value: str) -> str:
    """Normalize preference values enough to avoid duplicate spelling variants."""
    return re.sub(r"\s+", " ", value.strip().lower())


def _normalize_topic_name(value: str) -> str:
    """Normalize topic labels while preserving readable capitalization."""
    return re.sub(r"\s+", " ", value.strip())


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


def _has_balanced_single_quotes(text: str) -> bool:
    """Return False when a Cypher string literal was obviously truncated."""
    escaped = False
    in_quote = False

    for char in text:
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "'" and not escaped:
            in_quote = not in_quote
        escaped = False

    return not in_quote


def _cypher_uses_known_schema(cypher: str, schema: dict[str, set[str]]) -> bool:
    """Block generated reads that reference schema elements absent from Neo4j."""
    labels = set(re.findall(r"(?<!\[):([A-Za-z_][A-Za-z0-9_]*)\b", cypher))
    rel_types = set(re.findall(r"\[:([A-Za-z_][A-Za-z0-9_]*)\b", cypher))
    dotted_props = set(re.findall(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)\b", cypher))
    map_props = set(re.findall(r"[\{\s,]([A-Za-z_][A-Za-z0-9_]*)\s*:", cypher))
    property_keys = dotted_props | map_props

    return (
        labels <= schema["labels"]
        and rel_types <= schema["relationships"]
        and property_keys <= schema["properties"]
    )


def _is_safe_merge_cypher(cypher: str) -> bool:
    """Allow only the narrow MERGE shapes used by the entity extractor."""
    if not _has_balanced_single_quotes(cypher):
        return False

    normalized = re.sub(r"\s+", " ", cypher.strip()).upper()
    if not normalized.startswith("MERGE "):
        return False

    forbidden = {
        "CALL",
        "CREATE",
        "DELETE",
        "DETACH",
        "DROP",
        "FOREACH",
        "LOAD CSV",
        "MATCH",
        "OPTIONAL MATCH",
        "REMOVE",
        "RETURN",
        "SET",
        "UNWIND",
        "WITH",
    }
    if any(re.search(rf"\b{term}\b", normalized) for term in forbidden):
        return False

    allowed_labels = {"PERSON", "PREFERENCE", "SESSION", "TOPIC"}
    labels = set(re.findall(r"(?<!\[):([A-Z_][A-Z0-9_]*)\b", normalized))
    rel_types = set(re.findall(r"\[:([A-Z_][A-Z0-9_]*)\b", normalized))
    allowed_rel_types = {"ASKED_ABOUT", "HAS_PREFERENCE", "MENTIONS", "RELATED_TO"}

    return labels <= allowed_labels and rel_types <= allowed_rel_types


def _extract_merge_statements(text: str) -> List[str]:
    """Extract safe one-line MERGE statements from an LLM response."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:cypher)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    statements: List[str] = []
    for line in cleaned.splitlines():
        statement = line.strip().rstrip(";")
        if not statement or not statement.upper().startswith("MERGE "):
            continue
        if _is_safe_merge_cypher(statement):
            statements.append(statement)
        else:
            logger.debug("Cypher MERGE blocked by safety guard: %s", statement)

    return statements


def _extract_structured_facts(text: str) -> dict[str, Any]:
    """Parse and normalize the structured extraction payload."""
    try:
        payload = _clean_json(text)
    except Exception as e:
        logger.debug("Structured fact extraction JSON parse failed: %s", e)
        return {}

    facts: dict[str, Any] = {}

    person = payload.get("person")
    if isinstance(person, dict):
        name = person.get("name")
        if isinstance(name, str) and name.strip():
            facts["person_name"] = name.strip()
        age = person.get("age")
        if isinstance(age, int) and 0 < age < 130:
            facts["person_age"] = age
        elif isinstance(age, str) and age.isdigit():
            parsed_age = int(age)
            if 0 < parsed_age < 130:
                facts["person_age"] = parsed_age

    preferences = []
    for item in payload.get("preferences", []):
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        value = item.get("value")
        if not isinstance(key, str) or not isinstance(value, str):
            continue

        normalized_key = _normalize_preference_key(key)
        normalized_value = _normalize_preference_value(value)
        if normalized_key and normalized_value:
            preferences.append(
                {
                    "id": f"preference:{normalized_key}:{_slug(normalized_value)}",
                    "key": normalized_key,
                    "value": normalized_value,
                }
            )

    if preferences:
        deduped_preferences = {
            (item["key"], item["value"]): item
            for item in preferences
        }
        facts["preferences"] = list(deduped_preferences.values())

    topics = []
    for item in payload.get("topics", []):
        if isinstance(item, str):
            topic = _normalize_topic_name(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            topic = _normalize_topic_name(item["name"])
        else:
            continue

        if topic:
            topics.append({"id": f"topic:{_slug(topic)}", "name": topic})

    if topics:
        deduped_topics = {item["id"]: item for item in topics}
        facts["topics"] = list(deduped_topics.values())

    return facts


def _extract_deterministic_facts(text: str) -> dict[str, Any]:
    """Extract simple high-confidence facts without spending an LLM call."""
    facts: dict[str, Any] = {}

    age_match = re.search(r"\b(?:eu\s+)?tenho\s+(\d{1,3})\s+anos?\b", text.lower())
    if age_match:
        age = int(age_match.group(1))
        if 0 < age < 130:
            facts["person_age"] = age

    name_match = re.search(r"\b(?:meu nome é|me chamo|eu sou)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ' -]{1,60})", text, re.IGNORECASE)
    if name_match:
        facts["person_name"] = name_match.group(1).strip(" .,;:!?")

    return facts


def _merge_fact_dicts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge deterministic and LLM facts without duplicating list entries."""
    merged = dict(base)
    for key, value in extra.items():
        if key in {"preferences", "topics"} and isinstance(value, list):
            current = merged.get(key, [])
            if not isinstance(current, list):
                current = []
            by_id = {
                item.get("id", str(item)): item
                for item in [*current, *value]
                if isinstance(item, dict)
            }
            merged[key] = list(by_id.values())
        elif value is not None:
            merged[key] = value
    return merged


def _looks_like_persistable_fact(text: str) -> bool:
    """Cheap gate to avoid invoking extraction on ordinary conversational turns."""
    normalized = text.lower()
    cues = {
        "eu sou",
        "me chamo",
        "meu nome",
        "minha profissão",
        "minha profissao",
        "moro em",
        "trabalho em",
        "gosto de",
        "não gosto de",
        "nao gosto de",
        "prefiro",
        "favorito",
        "favorita",
        "lembre",
        "lembrar",
        "minha preferência",
        "minha preferencia",
        "tenho",
        "eu tenho",
        "anos",
    }
    return any(cue in normalized for cue in cues)


def _format_knowledge_rows(rows: List[dict[str, Any]], max_chars: int = 800) -> str:
    """Renders Neo4j rows compactly for injection into memory_context."""
    rendered = str(rows)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + "..."
    return "Knowledge Graph:\n" + rendered


def _get_knowledge_graph_schema() -> dict[str, set[str]]:
    """Read the current Neo4j schema without emitting missing-label warnings."""
    driver = get_neo4j_driver()
    with driver.session(database=None) as session:
        labels = session.run("CALL db.labels() YIELD label RETURN collect(label) AS labels").single()
        relationships = session.run(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN collect(relationshipType) AS relationships"
        ).single()
        properties = session.run(
            "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS properties"
        ).single()

    return {
        "labels": set(labels["labels"] if labels else []),
        "relationships": set(relationships["relationships"] if relationships else []),
        "properties": set(properties["properties"] if properties else []),
    }


def _persist_structured_facts(
    facts: dict[str, Any],
    session_id: str,
    user_input: str,
) -> None:
    """Persist normalized facts with canonical node IDs and connected relationships."""
    if not facts:
        return

    timestamp = datetime.now(timezone.utc).isoformat()
    person_name = facts.get("person_name") or DEFAULT_PERSON_NAME
    person_age = facts.get("person_age")
    aliases = sorted({DEFAULT_PERSON_NAME, person_name, "user", "usuario", "usuário"})

    driver = get_neo4j_driver()
    with driver.session(database=None) as session:
        session.run(
            """
            MERGE (p:Person {id: $person_id})
            SET p.name = $person_name,
                p.aliases = $aliases
            """,
            person_id=DEFAULT_PERSON_ID,
            person_name=person_name,
            aliases=aliases,
        ).consume()

        if isinstance(person_age, int):
            session.run(
                """
                MATCH (p:Person {id: $person_id})
                SET p.age = $person_age
                """,
                person_id=DEFAULT_PERSON_ID,
                person_age=person_age,
            ).consume()

        for preference in facts.get("preferences", []):
            session.run(
                """
                MATCH (p:Person {id: $person_id})
                MERGE (pref:Preference {id: $preference_id})
                SET pref.key = $key,
                    pref.value = $value
                MERGE (p)-[:HAS_PREFERENCE]->(pref)
                """,
                person_id=DEFAULT_PERSON_ID,
                preference_id=preference["id"],
                key=preference["key"],
                value=preference["value"],
            ).consume()

        topics = facts.get("topics", [])
        if topics:
            summary = user_input[:160]
            session.run(
                """
                MERGE (s:Session {id: $session_id})
                SET s.summary = $summary,
                    s.timestamp = coalesce(s.timestamp, $timestamp)
                """,
                session_id=session_id,
                summary=summary,
                timestamp=timestamp,
            ).consume()

        for topic in topics:
            session.run(
                """
                MATCH (s:Session {id: $session_id})
                MERGE (t:Topic {id: $topic_id})
                SET t.name = $name
                MERGE (s)-[:MENTIONS]->(t)
                """,
                session_id=session_id,
                topic_id=topic["id"],
                name=topic["name"],
            ).consume()


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

    try:
        schema = _get_knowledge_graph_schema()
    except Exception as e:
        logger.warning("Knowledge graph schema lookup failed: %s", e)
        return {}

    if not schema["labels"] or not schema["relationships"]:
        logger.debug("Knowledge graph is empty; skipping Cypher retrieval.")
        return {}

    lowered_query = query.lower()
    available_terms = {
        value.lower()
        for values in schema.values()
        for value in values
    }
    available_terms.update(
        {
            "animal",
            "favorito",
            "favorita",
            "preferencia",
            "preferência",
            "gosto",
            "lembra",
            "memoria",
            "memória",
        }
    )
    if not any(term in lowered_query for term in available_terms):
        logger.debug("Question does not appear to target the knowledge graph; skipping Cypher retrieval.")
        return {}

    schema_lines = [
        f"- Labels currently present: {', '.join(sorted(schema['labels']))}",
        f"- Relationship types currently present: {', '.join(sorted(schema['relationships']))}",
    ]
    if schema["properties"]:
        schema_lines.append(f"- Property keys currently present: {', '.join(sorted(schema['properties']))}")

    llm = get_llm()
    text2cypher_prompt = f"""
You are a Neo4j Cypher specialist. Given this knowledge graph schema:
- (:Person {{id, name, aliases}})-[:HAS_PREFERENCE]->(:Preference {{id, key, value}})
- (:Session {{id, summary, timestamp}})-[:MENTIONS]->(:Topic {{id, name}})
- (:Topic {{id, name}})-[:RELATED_TO]->(:Topic {{id, name}})

Current database schema snapshot:
{chr(10).join(schema_lines)}

IMPORTANT RULES:
1. Use only labels, relationship types, and property keys that exist in the current database snapshot.
2. The current user is Person id '{DEFAULT_PERSON_ID}', name '{DEFAULT_PERSON_NAME}'. Prefer p.id = '{DEFAULT_PERSON_ID}' over p.name.
3. Only use MATCH, OPTIONAL MATCH, WITH, RETURN, ORDER BY, LIMIT, and read-only CALL clauses.
4. Never use CREATE, MERGE, SET, DELETE, DETACH, REMOVE, DROP, or LOAD CSV.
5. If the current schema cannot answer the question, return: RETURN null AS result LIMIT 0
6. Return only the Cypher query, with no explanations or Markdown.

Question: {query}
""".strip()

    try:
        response = llm.invoke([HumanMessage(content=text2cypher_prompt)])
        cypher_query = _clean_cypher(str(response.content))

        if not _is_safe_readonly_cypher(cypher_query):
            logger.debug("Cypher query blocked by read-only guard: %s", cypher_query)
            return {}

        if not _cypher_uses_known_schema(cypher_query, schema):
            logger.debug("Cypher query blocked by schema guard: %s", cypher_query)
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


def entity_extraction_node(state: ConversationState) -> dict:
    """
    Extracts stable facts from the turn and persists them in Neo4j.

    The extractor accepts only one-line MERGE statements in the graph schema used
    by knowledge_retrieval_node, keeping writes idempotent and tightly scoped.
    """
    if not memory_settings.MEMORY_ENABLED or not memory_settings.KNOWLEDGE_GRAPH_ENABLED:
        return {}

    if not state.get("should_write_memory"):
        return {}

    user_input = state.get("user_input", "")
    agent_response = state.get("response", "")
    session_id = state.get("session_id") or "unknown"

    if not user_input.strip() or not agent_response.strip():
        return {}

    if not _looks_like_persistable_fact(user_input):
        logger.debug("Turn does not look like a persistable fact; skipping entity extraction.")
        return {}

    deterministic_facts = _extract_deterministic_facts(user_input)
    if deterministic_facts and not any(
        cue in user_input.lower()
        for cue in ("gosto", "prefiro", "favorito", "favorita", "lembre", "lembrar", "sobre")
    ):
        try:
            _persist_structured_facts(
                facts=deterministic_facts,
                session_id=session_id,
                user_input=user_input,
            )
        except Exception as e:
            logger.warning("Deterministic entity persistence failed: %s", e)
        return {}

    llm = get_llm()
    extraction_prompt = NisabaPrompts.get_entity_extraction_prompt(
        user_input=user_input,
        agent_response=agent_response,
        session_id=session_id,
    )

    try:
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        facts = _merge_fact_dicts(deterministic_facts, _extract_structured_facts(str(response.content)))

        if not facts:
            logger.debug("No structured facts extracted for this turn.")
            return {}

        _persist_structured_facts(
            facts=facts,
            session_id=session_id,
            user_input=user_input,
        )

    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)

    return {}


def conversation_node(state: ConversationState) -> dict:
    """Core reasoning node: invokes LLM with personality and context."""
    llm = get_llm()
    raw_messages = state.get("messages", [])
    user_input = state.get("user_input", "")
    messages = [message for message in raw_messages if not isinstance(message, SystemMessage)]

    if messages and isinstance(messages[-1], HumanMessage) and messages[-1].content == user_input:
        conversation_history = messages
    else:
        conversation_history = messages + [HumanMessage(content=user_input)]

    # Retrieve centralized prompt from prompt.py
    system_prompt = NisabaPrompts.get_main_system_prompt(memory_context=state.get("memory_context"))

    llm_messages = [SystemMessage(content=system_prompt)] + conversation_history

    try:
        response = llm.invoke(llm_messages)
        return {
            "messages": conversation_history + [response],
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

"""End-to-end test of memory integration."""

import sys
from pathlib import Path
from dotenv import load_dotenv  # ← Adicionar

# 🔹 Carregar o .env global ANTES de qualquer import do pacote nisaba
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✅ .env carregado: {env_path}")

# Adicionar src/ e a raiz do repo ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Agora importar o factory
from agents.shared.config.memory import memory_settings
from nisaba.agent.graph import build_conversation_graph
from nisaba.agent.graph_factory import get_conversation_graph
from nisaba.agent.node import (
    _extract_deterministic_facts,
    _extract_merge_statements,
    _extract_structured_facts,
    _looks_like_persistable_fact,
)
from nisaba.memory.vector_store import _safe_text_for_bm25_query


def test_shared_memory_config_imports_from_global_env() -> None:
    assert memory_settings.DATABASE_URL
    assert "noosphera_agents_db" in memory_settings.DATABASE_URL


def test_conversation_graph_builds() -> None:
    graph = build_conversation_graph()
    assert sorted(graph.nodes.keys()) == [
        "conversation",
        "entity_extract",
        "input",
        "knowledge_retrieve",
        "memory_retrieve",
        "memory_write",
    ]


def test_entity_extraction_rejects_malformed_cypher() -> None:
    malformed = (
        "MERGE (:Person {name: 'Yuri})-[:HAS_PREFERENCE]->"
        "(:Preference {key: 'favorite_animal', value: 'gato'})"
    )
    assert _extract_merge_statements(malformed) == []


def test_entity_extraction_fact_gate() -> None:
    assert _looks_like_persistable_fact("Tenho 37 anos")
    assert _looks_like_persistable_fact("Meu animal favorito é gato")
    assert not _looks_like_persistable_fact("Pode explicar melhor essa ideia?")


def test_structured_extraction_normalizes_preferences() -> None:
    facts = _extract_structured_facts(
        '{"person": {"name": "Yuri", "age": 37}, '
        '"preferences": [{"key": "animal", "value": "Gato"}], '
        '"topics": [{"name": "Neo4j"}]}'
    )

    assert facts["person_name"] == "Yuri"
    assert facts["person_age"] == 37
    assert facts["preferences"] == [
        {
            "id": "preference:favorite_animal:gato",
            "key": "favorite_animal",
            "value": "gato",
        }
    ]
    assert facts["topics"] == [{"id": "topic:neo4j", "name": "Neo4j"}]


def test_deterministic_extraction_gets_age() -> None:
    assert _extract_deterministic_facts("Tenho 37 anos") == {"person_age": 37}


def test_bm25_query_sanitizes_structured_text() -> None:
    query = _safe_text_for_bm25_query(
        'Bom refatorei você:\n'
        'Person sempre vira (:Person {id: "local_user", name: "Yuri"})\n'
        'preferência vira preference:favorite_animal:gato'
    )

    assert ":" not in query
    assert "{" not in query
    assert "}" not in query
    assert " OR " in query


def run_memory_flow() -> None:
    print("🧪 Testing memory flow...")

    # Test 1: Basic conversation
    with get_conversation_graph(session_id="test-001") as graph:
        result = graph.invoke(
            {
                "messages": [],
                "user_input": "Olá, qual é o seu nome?",
                "session_id": "test-001",
                "response": "",
            },
            config={"configurable": {"thread_id": "test-001"}},
        )
        print(f"✅ Response: {result.get('response', 'No response')[:100]}...")

    # Test 2: Memory retrieval (if enabled)
    if memory_settings.MEMORY_ENABLED:
        with get_conversation_graph(session_id="test-001") as graph:
            result2 = graph.invoke(
                {
                    "messages": [],
                    "user_input": "Lembra da nossa conversa anterior?",
                    "session_id": "test-001",
                    "response": "",
                },
                config={"configurable": {"thread_id": "test-001"}},
            )
            print(f"✅ Memory context: {'Yes' if result2.get('memory_context') else 'No'}")

    print("🎉 Memory flow test complete!")


if __name__ == "__main__":
    run_memory_flow()

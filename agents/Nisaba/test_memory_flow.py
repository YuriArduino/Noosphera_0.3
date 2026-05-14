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


def test_shared_memory_config_imports_from_global_env() -> None:
    assert memory_settings.DATABASE_URL
    assert "noosphera_agents_db" in memory_settings.DATABASE_URL


def test_conversation_graph_builds() -> None:
    graph = build_conversation_graph()
    assert sorted(graph.nodes.keys()) == [
        "conversation",
        "input",
        "memory_retrieve",
        "memory_write",
    ]


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

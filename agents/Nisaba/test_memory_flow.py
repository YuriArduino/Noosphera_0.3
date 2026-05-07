"""End-to-end test of memory integration."""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv  # ← Adicionar

# 🔹 Carregar .env ANTES de qualquer import do pacote nisaba
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✅ .env carregado: {env_path}")

# Adicionar src/ ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))


# Agora importar o factory
from nisaba.agent.factory import get_conversation_graph
from nisaba.config.memory import memory_settings

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

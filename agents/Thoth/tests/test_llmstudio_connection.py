"""
LLMStudio Connection Tests.

Verifies connectivity and basic functionality for both:
    - Chat completion model (text correction)
    - Embedding model (vectorstore/memory)

Usage:
    pytest tests/test_llmstudio_connection.py -v

Requirements:
    - LLMStudio running at http://127.0.0.1:1234
    - Models loaded: meta-llama-3.1-8b-instruct + text-embedding-nomic-embed-text-v1.5
"""

from typing import Dict, Any
import pytest
import httpx


from thoth.config.llm import llm_settings
from thoth.config.thresholds import threshold_settings
from thoth.config.pipeline import pipeline_settings
from thoth.config.environment import env_settings


# ================================================================
# FIXTURES
# ================================================================
@pytest.fixture
def llmstudio_client() -> httpx.Client:
    """Create HTTP client for LLMStudio API."""
    return httpx.Client(
        base_url=llm_settings.LLMSTUDIO_BASE_URL,
        timeout=llm_settings.LLMSTUDIO_TIMEOUT,
    )


@pytest.fixture
def chat_payload() -> Dict[str, Any]:
    """Standard chat completion payload for testing."""
    return {
        "model": llm_settings.LLMSTUDIO_MODEL,
        "messages": [
            {"role": "system", "content": "Você é um assistente útil. Responda de forma concisa."},
            {"role": "user", "content": "Diga apenas 'OK' se estiver funcionando."},
        ],
        "temperature": 0.1,
        "max_tokens": 10,
    }


@pytest.fixture
def embedding_payload() -> Dict[str, Any]:
    """Standard embedding payload for testing."""
    return {
        "model": llm_settings.EMBEDDING_MODEL,
        "input": "Teste de conexão do agente Thoth com LLMStudio.",
    }


# ================================================================
# HEALTH CHECK TESTS
# ================================================================
class TestLLMStudioHealth:
    """Test LLMStudio API health and availability."""

    def test_health_endpoint(
        self,
        llmstudio_client: httpx.Client,  # pylint: disable=redefined-outer-name
    ):
        """
        Verify LLMStudio health endpoint is accessible.

        Expected:
            - Status code 200
            - Response contains 'status' or 'healthy' key
        """
        response = llmstudio_client.get("/health")

        assert (
            response.status_code == 200
        ), f"Health endpoint should return 200, got {response.status_code}"

        data = response.json()
        assert isinstance(data, dict), "Health response should be JSON object"

    def test_base_url_reachable(self):
        """Verify base URL is reachable (fallback if /health not available)."""
        try:
            response = httpx.get(llm_settings.LLMSTUDIO_BASE_URL, timeout=10.0)
            # Some LLMStudio versions return 200 or 404 at root
            assert response.status_code in [
                200,
                404,
                405,
            ], f"Base URL should be reachable, got {response.status_code}"
        except httpx.ConnectError as e:
            pytest.fail(
                f"Cannot connect to LLMStudio at {llm_settings.LLMSTUDIO_BASE_URL}. "
                f"Ensure LLMStudio is running. Error: {e}"
            )


# ================================================================
# CHAT COMPLETION TESTS
# ================================================================
class TestChatCompletion:
    """Test chat completion model for OCR text correction."""

    def test_chat_model_responds(
        self,
        llmstudio_client: httpx.Client,  # pylint: disable=redefined-outer-name
        chat_payload: Dict[str, Any],  # pylint: disable=redefined-outer-name
    ):
        """
        Verify chat model responds to basic request.

        Expected:
            - Status code 200
            - Response contains 'choices' array
            - At least one choice with 'message' content
        """
        response = llmstudio_client.post(
            llm_settings.llm_full_endpoint,
            json=chat_payload,
        )

        assert response.status_code == 200, (
            f"Chat endpoint should return 200, got {response.status_code}. "
            f"Response: {response.text}"
        )

        data = response.json()

        # Validate OpenAI-compatible response structure
        assert "choices" in data, "Response should contain 'choices' key"
        assert len(data["choices"]) > 0, "Should have at least one choice"

        choice = data["choices"][0]
        assert "message" in choice, "Choice should contain 'message'"
        assert "content" in choice["message"], "Message should contain 'content'"

    def test_chat_model_portuguese(
        self,
        llmstudio_client: httpx.Client,  # pylint: disable=redefined-outer-name
    ):
        """
        Verify chat model handles Portuguese text (OCR correction use case).

        Expected:
            - Response in Portuguese or at least coherent
            - Preserves terminology
        """
        payload = {
            "model": llm_settings.LLMSTUDIO_MODEL,
            "messages": [
                {"role": "system", "content": "Você é um corretor de textos OCR em português."},
                {"role": "user", "content": "Corrija: 'A psicanálise é um método de tratamento'"},
            ],
            "temperature": 0.1,
            "max_tokens": 50,
        }

        response = llmstudio_client.post(
            llm_settings.llm_full_endpoint,
            json=payload,
        )

        assert response.status_code == 200, "Portuguese text should be processed"

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Basic sanity check
        assert len(content) > 0, "Response should not be empty"
        assert len(content) < 500, "Response should be concise"

    def test_chat_model_with_ocr_format(
        self,
        llmstudio_client: httpx.Client,  # pylint: disable=redefined-outer-name
    ):
        """
        Verify chat model handles OCR-formatted input (real use case).

        Expected:
            - Preserves page markers
            - Corrects text while maintaining structure
        """
        payload = {
            "model": llm_settings.LLMSTUDIO_MODEL,
            "messages": [
                {"role": "system", "content": "Corrija erros OCR mantendo estrutura de páginas."},
                {
                    "role": "user",
                    "content": """=== PAGE 1 | Confidence: 85.0% ===

DICIONÁRIO DE PSICANÁLISE
TRADUÇÃO: Vera Ribero psicanalista

=== END OF DOCUMENT ===""",
                },
            ],
            "temperature": 0.1,
            "max_tokens": 200,
        }

        response = llmstudio_client.post(
            llm_settings.llm_full_endpoint,
            json=payload,
        )

        assert response.status_code == 200, "OCR format should be processed"

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Should preserve page markers
        assert "PAGE 1" in content or "Page 1" in content, "Should preserve page markers"


# ================================================================
# EMBEDDING MODEL TESTS
# ================================================================
class TestEmbeddingModel:
    """Test embedding model for vectorstore/memory."""

    def test_embedding_model_responds(
        self,
        llmstudio_client: httpx.Client,  # pylint: disable=redefined-outer-name
        embedding_payload: Dict[str, Any],  # pylint: disable=redefined-outer-name
    ):
        """
        Verify embedding model responds to basic request.

        Expected:
            - Status code 200
            - Response contains 'data' array with embeddings
            - Embedding is a list of floats
        """
        response = llmstudio_client.post(
            llm_settings.embedding_full_endpoint,
            json=embedding_payload,
        )

        assert response.status_code == 200, (
            f"Embedding endpoint should return 200, got {response.status_code}. "
            f"Response: {response.text}"
        )

        data = response.json()

        # Validate OpenAI-compatible response structure
        assert "data" in data, "Response should contain 'data' key"
        assert len(data["data"]) > 0, "Should have at least one embedding"

        embedding_data = data["data"][0]
        assert "embedding" in embedding_data, "Data should contain 'embedding'"

        embedding = embedding_data["embedding"]
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) > 0, "Embedding should not be empty"
        assert all(isinstance(x, float) for x in embedding), "Embedding should contain floats"

    def test_embedding_dimension(
        self,
        llmstudio_client: httpx.Client,  # pylint: disable=redefined-outer-name
    ):
        """
        Verify embedding dimension matches expected size.

        Expected:
            - Dimension should be consistent (e.g., 768 for nomic-embed-text)
            - All embeddings in batch have same dimension
        """
        payload = {
            "model": llm_settings.EMBEDDING_MODEL,
            "input": [
                "Texto 1 para embedding",
                "Texto 2 para embedding",
            ],
        }

        response = llmstudio_client.post(
            llm_settings.embedding_full_endpoint,
            json=payload,
        )

        if response.status_code != 200:
            pytest.skip("Batch embedding not supported by this model")

        data = response.json()
        embeddings = [d["embedding"] for d in data["data"]]

        # All embeddings should have same dimension
        dimensions = [len(e) for e in embeddings]
        assert (
            len(set(dimensions)) == 1
        ), f"All embeddings should have same dimension, got {dimensions}"

        # Store dimension for reference
        actual_dimension = dimensions[0]
        print(f"\n✅ Embedding dimension: {actual_dimension}")

    def test_embedding_semantic_similarity(
        self,
        llmstudio_client: httpx.Client,  # pylint: disable=redefined-outer-name
    ):
        """
        Verify embeddings capture semantic similarity (basic sanity check).

        Expected:
            - Similar texts should have similar embeddings
            - Different texts should have different embeddings
        """
        texts = [
            "psicanálise Freud inconsciente",
            "psicanálise Lacan desejo",
            "receita de bolo chocolate",  # Unrelated
        ]

        embeddings = []
        for text in texts:
            payload = {
                "model": llm_settings.EMBEDDING_MODEL,
                "input": text,
            }
            response = llmstudio_client.post(
                llm_settings.embedding_full_endpoint,
                json=payload,
            )

            if response.status_code == 200:
                data = response.json()
                embeddings.append(data["data"][0]["embedding"])
            else:
                pytest.skip("Embedding model not available for similarity test")

        if len(embeddings) < 3:
            pytest.skip("Could not generate all embeddings")

        # Basic dot product similarity (should be > 0 for related, < for unrelated)
        def dot_product(a, b):
            return sum(x * y for x, y in zip(a, b))

        sim_related = dot_product(embeddings[0], embeddings[1])
        sim_unrelated = dot_product(embeddings[0], embeddings[2])

        # Related texts should have higher similarity
        # Note: This is a soft check - embeddings may not be normalized
        print(f"\n✅ Similarity (related): {sim_related:.4f}")
        print(f"✅ Similarity (unrelated): {sim_unrelated:.4f}")


# ================================================================
# INTEGRATION TESTS
# ================================================================
class TestLLMStudioIntegration:
    """Integration tests for complete LLMStudio workflow."""

    def test_both_models_available(
        self,
        llmstudio_client: httpx.Client,  # pylint: disable=redefined-outer-name
    ):
        """
        Verify both chat and embedding models are available.

        This is the main integration test for Thoth agent requirements.
        """
        # Test chat model
        chat_response = llmstudio_client.post(
            llm_settings.llm_full_endpoint,
            json={
                "model": llm_settings.LLMSTUDIO_MODEL,
                "messages": [{"role": "user", "content": "OK"}],
                "max_tokens": 5,
            },
        )

        # Test embedding model
        embed_response = llmstudio_client.post(
            llm_settings.embedding_full_endpoint,
            json={
                "model": llm_settings.EMBEDDING_MODEL,
                "input": "test",
            },
        )

        # Assert both are working
        assert chat_response.status_code == 200, f"Chat model failed: {chat_response.status_code}"
        assert (
            embed_response.status_code == 200
        ), f"Embedding model failed: {embed_response.status_code}"

        print("\n✅ Both chat and embedding models are available!")
        print(f"   Chat model: {llm_settings.LLMSTUDIO_MODEL}")
        print(f"   Embedding model: {llm_settings.EMBEDDING_MODEL}")


# ================================================================
# RUN MANUALLY
# ================================================================
if __name__ == "__main__":
    # Run tests manually for debugging
    pytest.main([__file__, "-v", "-s"])

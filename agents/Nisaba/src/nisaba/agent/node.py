"""
Nisaba Agent Nodes — LangGraph Execution Units with Memory.
Stage 2: Hybrid Memory Integration.
"""

import logging
import re
import uuid
from typing import List, Optional, TypedDict
from datetime import datetime, timezone
from functools import lru_cache

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from nisaba.config.llm import llm_settings
from nisaba.config.memory import memory_settings
from nisaba.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

# =============================================================================
# STATE DEFINITION
# =============================================================================


class ConversationState(TypedDict):
    messages: List[BaseMessage]
    user_input: str
    response: str
    session_id: Optional[str]
    memory_context: Optional[str]
    should_write_memory: bool


# =============================================================================
# SINGLETON / CACHED INSTANCES
# =============================================================================

vector_store = VectorStore()


@lru_cache(maxsize=1)
def _get_llm():
    return ChatOpenAI(
        model=llm_settings.CHAT_MODEL,
        api_key=llm_settings.LLM_API_KEY,
        base_url=llm_settings.LLM_BASE_URL,
        temperature=llm_settings.CHAT_TEMPERATURE,
        max_completion_tokens=llm_settings.CHAT_MAX_TOKENS,
        timeout=llm_settings.LLM_TIMEOUT,
        max_retries=llm_settings.LLM_MAX_RETRIES,
    )


# =============================================================================
# UTILITY
# =============================================================================


def clean_reasoning(text: str) -> str:
    """Remove o conteúdo dentro de tags <thought> ou <think>."""
    return re.sub(r"<(thought|think)>.*?</\1>", "", text, flags=re.DOTALL).strip()


# =============================================================================
# CORE NODES
# =============================================================================


def input_node(state: ConversationState) -> dict:
    user_input = state.get("user_input", "").strip()
    if not user_input:
        return {"response": "Por favor, envie uma mensagem válida."}
    if not state.get("session_id"):
        state["session_id"] = str(uuid.uuid4())[:8]
    return {"user_input": user_input, "session_id": state["session_id"]}


def memory_retrieval_node(state: ConversationState) -> dict:
    if not memory_settings.MEMORY_ENABLED or not memory_settings.VECTORSTORE_ENABLED:
        return {}
    query = state.get("user_input", "")
    if len(query.strip()) < 10:
        return {}
    try:
        similar = vector_store.search_similar(
            query=query,
            limit=memory_settings.SEMANTIC_SEARCH_TOP_K,
            min_relevance=0.3,
        )
        if similar:
            context_snippets = [
                f"[Experiência #{i+1}] {exp.title or 'Sem título'}: {exp.content[:200]}..."
                for i, exp in enumerate(similar[:3])
            ]
            memory_context = "\n\n".join(context_snippets)
            top_exp = similar[0]
            if top_exp.id is not None:
                vector_store.increment_usage(top_exp.id)
            return {"memory_context": memory_context}
    except Exception as e:
        logger.warning("Memory retrieval failed: %s", e)
    return {}


def conversation_node(state: ConversationState) -> dict:
    llm = _get_llm()
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")

    system_prompt = (
        "Você é Nisaba, um assistente pessoal com memória persistente. "
        "Analise o contexto de memória antes de responder. "
        "Pense passo a passo sobre como as informações passadas"
        " se conectam com a pergunta atual. "
        "Escreva seu raciocínio dentro de tags <thought>"
        " e depois forneça a resposta final."
    )

    memory_context = state.get("memory_context")
    if memory_context:
        system_prompt += f"\n\n🧠 Contexto:\n{memory_context}"

    llm_messages = (
        [SystemMessage(content=system_prompt)] + messages + [HumanMessage(content=user_input)]
    )

    try:
        response = llm.invoke(llm_messages)
        # Remove o raciocínio para exibição e armazenamento
        clean_response = clean_reasoning(response.content)
        return {
            "messages": llm_messages + [response],
            "response": clean_response,
            "should_write_memory": True,
        }
    except Exception as e:
        logger.error("LLM invocation failed: %s", e)
        return {
            "response": f"⚠️ Erro ao processar: {type(e).__name__}",
            "should_write_memory": False,
        }


# =============================================================================
# REFLECTION NODE (desabilitado temporariamente)
# =============================================================================
# def reflection_node(state: ConversationState) -> dict:
#     """
#     Refina a resposta com base no contexto de memória, garantindo consistência.
#     """
#     draft = state.get("response", "")
#     memory_context = state.get("memory_context")

#     # Se não há contexto de memória, não precisa refinar
#     if not memory_context:
#         return {"response": draft, "should_write_memory": True}

#     llm = _get_llm()

#     system_prompt = (
#         "Você é um revisor de respostas. Você receberá:\n"
#         "1. Um contexto de memória com informações que o assistente já sabe.\n"
#         "2. Um rascunho de resposta.\n\n"
#         "Se o rascunho disser que não sabe ou que não pode acessar informações, "
#         "mas o contexto contém a resposta, REWRITE a resposta fornecendo a informação correta. "
#         "Exemplo: se o contexto diz que o usuário se chama Yuri e o rascunho diz 'não sei seu nome', "
#         "a resposta final deve ser 'Você se chama Yuri, como me disse antes'.\n\n"
#         "Seja direto. NÃO repita que não pode acessar dados quando o contexto já os fornece."
#     )

#     user_message = (
#         f"Contexto (use estes dados):\n{memory_context}\n\n"
#         f"Rascunho a revisar:\n{draft}\n\n"
#         "Responda APENAS com a resposta final corrigida. "
#         "Se o rascunho estiver OK, copie-o."
#     )

#     try:
#         response = llm.invoke(
#             [
#                 SystemMessage(content=system_prompt),
#                 HumanMessage(content=user_message),
#             ]
#         )
#         final_response = response.content.strip()
#         return {
#             "response": final_response,
#             "should_write_memory": True,  # já decidido no conversation_node
#         }
#     except Exception as e:
#         logger.warning("Reflection failed, usando rascunho original: %s", e)
#         return {"response": draft, "should_write_memory": True}


def memory_writer_node(state: ConversationState) -> dict:
    if not state.get("should_write_memory"):
        return {}
    if not memory_settings.MEMORY_ENABLED or not memory_settings.VECTORSTORE_ENABLED:
        return {}
    try:
        user_input = state.get("user_input", "")
        agent_response = state.get("response", "")
        content = f"Usuário: {user_input}\nNisaba: {agent_response}"
        vector_store.add_experience(
            content=content,
            session_id=state.get("session_id") or "unknown",
            title=f"Interação: {user_input[:50]}...",
            category="conversation",
            tags=["user_query", "agent_response"],
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tokens_estimated": len(content.split()) * 1.3,
            },
        )
    except Exception as e:
        logger.warning("Memory write failed: %s", e)
    return {}

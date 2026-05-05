"""
Thoth Agent Nodes — LangGraph Execution Units.
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode

from thoth.domain.state import ThothState
from thoth.domain.policy import ThothDecisionPolicy
from thoth.domain.decision import DecisionContext, QualityMetrics
from thoth.agent.tool import GlypharInfrastructureTool
from thoth.config import settings

# Instanciamos a ferramenta que criamos anteriormente
infra_tool = GlypharInfrastructureTool()
tool_node = ToolNode([infra_tool.glyphar_ocr_task])


async def node_ocr_execution(state: ThothState) -> Dict[str, Any]:
    """
    Nó que decide qual estratégia usar e chama a ferramenta do Prefect.
    """
    current_doc = state["documents"][0]  # Simplificado para 1 doc por vez
    attempt = state.get("reprocess_attempts", {}).get(current_doc, 1)

    # Thoth escolhe a estratégia baseado no histórico ou usa a inicial
    strategy = state.get("initial_strategy", settings.pipeline.INITIAL_STRATEGY)

    # Prepara a chamada da ferramenta
    message = HumanMessage(content=f"Process document {current_doc} with strategy {strategy}")

    # O LangGraph vai encaminhar isso para o ToolNode automaticamente
    return {"messages": [message], "current_step": "ocr_execution"}


async def node_analysis(state: ThothState) -> Dict[str, Any]:
    """
    Nó de reflexão: Lê o Postgres SST e aplica a ThothDecisionPolicy.
    """
    # 1. Recupera o último resultado da ferramenta (que já salvou no Postgres)
    # Aqui o agente "percebe" o que o Glyphar fez.
    last_tool_output = state["messages"][-1].content

    # 2. Avaliação de Doutrina
    # (Lógica simplificada: na vida real, extraímos o db_file_id do output)
    # decision = ThothDecisionPolicy.evaluate(context)

    return {"current_step": "decide"}

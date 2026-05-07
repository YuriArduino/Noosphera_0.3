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
from nisaba.config import settings

# Instanciamos a ferramenta que criamos anteriormente
infra_tool = GlypharInfrastructureTool()
tool_node = ToolNode([infra_tool.glyphar_ocr_task])

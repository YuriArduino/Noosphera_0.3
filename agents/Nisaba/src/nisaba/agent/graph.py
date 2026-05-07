"""
Thoth Strategic Graph — Noosphera 0.3.
Orchestrates Ingestion, Prefect Workers, and Quality Assessment.
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from thoth.domain.state import ThothState
from thoth.agent.nodes import agent_node, router_logic, analysis_node, finalization_node
from thoth.agent.tool import GlypharInfrastructureTool


def build_thoth_graph():
    # 1. Initialize State Graph
    workflow = StateGraph(ThothState)

    # 2. Define Tools Node (The Hands)
    infra_tool = GlypharInfrastructureTool()
    # thoth_tools = [infra_tool.glyphar_ocr_task, search_semantic_memory, ...]
    tool_node = ToolNode([infra_tool.glyphar_ocr_task])

    # 3. Add Nodes (The Brain)
    workflow.add_node("brain", agent_node)  # LLM Thinking

    # 4. Define Edges (The Flow)
    workflow.add_edge(START, "brain")

    # Conditional route: Tools or Assessment?
    workflow.add_conditional_edges(
        "brain", router_logic, {"call_tool": "infrastructure", "evaluate": "assessment"}
    )

    # After infra (Prefect), we always analyze the result in the SST
    workflow.add_edge("infrastructure", "assessment")

    # Assessment decides: Reprocess (back to brain) or Finish?
    def post_analysis_router(state: ThothState) -> Literal["retry", "end"]:
        # Se a Policy recomendou REPROCESS, volta para o cérebro
        if state["decisions"] and state["decisions"][-1]["action"] == "reprocess":
            return "retry"
        return "end"

    workflow.add_conditional_edges(
        "assessment", post_analysis_router, {"retry": "brain", "end": "finalize"}
    )

    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=checkpointer)

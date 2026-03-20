from typing import Literal

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.agent.policy_rag import policy_rag_graph
from src.agent.recommend import recommended_graph
from src.agent.reserve import reserve_graph
from src.agent.common.context import ContextSchema
from src.agent.node.main import (
    memory_manager_node,
    supervisor_router_node,
    empathy_generator_node,
)
from src.agent.state.main import State

builder = StateGraph(State, context_schema=ContextSchema)

builder.add_node("memory_manager", memory_manager_node)
builder.add_node("supervisor_router", supervisor_router_node)
builder.add_node("recommend_subgraph", recommended_graph)
builder.add_node("policy_rag_subgraph", policy_rag_graph)
builder.add_node("handoff_subgraph", reserve_graph)
builder.add_node("empathy_generator", empathy_generator_node)

builder.add_edge(START, "memory_manager")
builder.add_edge("memory_manager", "supervisor_router")


def router_message(state: State) -> Literal[
    "recommend_subgraph",
    "policy_rag_subgraph",
    "handoff_subgraph",
    "empathy_generator",
    "__end__",
]:
    destination = state.get("route_destination")
    if not destination:
        # 向后兼容旧 route 字段
        legacy = state.get("route", "handoff_subgraph")
        if legacy in {"recommend_subgraph", "property_rag_graph"}:
            destination = "RECOMMEND"
        elif legacy in {"policy_rag_subgraph", "policy_rag_graph"}:
            destination = "POLICY_RAG"
        elif legacy in {"handoff_subgraph", "reserve_graph", "extend_graph", "memory_manager_response"}:
            destination = "HANDOFF"
        elif legacy in {"empathy_generator"}:
            destination = "EMPATHY"
        elif legacy in {"__end__"}:
            destination = "END"
        else:
            destination = "HANDOFF"

    return {
        "RECOMMEND": "recommend_subgraph",
        "POLICY_RAG": "policy_rag_subgraph",
        "HANDOFF": "handoff_subgraph",
        "EMPATHY": "empathy_generator",
        "END": "__end__",
    }.get(destination, "handoff_subgraph")

builder.add_conditional_edges(
    "supervisor_router",
    router_message,
    {
        "recommend_subgraph": "recommend_subgraph",
        "policy_rag_subgraph": "policy_rag_subgraph",
        "handoff_subgraph": "handoff_subgraph",
        "empathy_generator": "empathy_generator",
        "__end__": END,
    },
)

# 控制权回传：所有 spoke 子图统一回到中心 supervisor
builder.add_edge("recommend_subgraph", "supervisor_router")
builder.add_edge("policy_rag_subgraph", "supervisor_router")
builder.add_edge("handoff_subgraph", "supervisor_router")

# 统一出口
builder.add_edge("empathy_generator", END)
graph = builder.compile()


def build_main_graph():
    return graph

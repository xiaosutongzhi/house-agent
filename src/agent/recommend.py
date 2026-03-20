from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.common.context import ContextSchema
from src.agent.node.recommend import (
    SQL_Search_Tool,
    Vector_Search_Tool,
    recommend_finalize_node,
    recommend_llm_node,
)
from src.agent.state.main import State

builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node("recommend_llm", recommend_llm_node)
builder.add_node("recommend_tools", ToolNode([SQL_Search_Tool, Vector_Search_Tool]))
builder.add_node("recommend_finalize", recommend_finalize_node)

builder.add_edge(START, "recommend_llm")
builder.add_conditional_edges(
    "recommend_llm",
    tools_condition,
    {
        "tools": "recommend_tools",
        "__end__": "recommend_finalize",
    },
)
builder.add_edge("recommend_tools", "recommend_llm")
builder.add_edge("recommend_finalize", END)

recommended_graph = builder.compile()


def build_recommend_subgraph():
    return recommended_graph





from langgraph.constants import START
from langgraph.graph import StateGraph

from src.agent.common.context import ContextSchema
from src.agent.node.reserve import handoff_reserve_node
from src.agent.state.main import State

builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node("handoff_reserve_node", handoff_reserve_node)
builder.add_edge(START, "handoff_reserve_node")
reserve_graph = builder.compile()


def build_handoff_subgraph():
    return reserve_graph

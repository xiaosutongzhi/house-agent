from langgraph.constants import START
from langgraph.graph import StateGraph

from src.agent.common.context import ContextSchema
from src.agent.node.policy_rag import (
    policy_generate_node,
    policy_rerank_node,
    policy_retrieve_node,
    policy_rewrite_node,
)
from src.agent.state.main import State

builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node("policy_rewrite", policy_rewrite_node)
builder.add_node("policy_retrieve", policy_retrieve_node)
builder.add_node("policy_rerank", policy_rerank_node)
builder.add_node("policy_generate", policy_generate_node)
builder.add_edge(START, "policy_rewrite")
builder.add_edge("policy_rewrite", "policy_retrieve")
builder.add_edge("policy_retrieve", "policy_rerank")
builder.add_edge("policy_rerank", "policy_generate")

policy_rag_graph = builder.compile()


def build_policy_rag_subgraph():
    return policy_rag_graph

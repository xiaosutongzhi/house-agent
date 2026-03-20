"""New LangGraph Agent.

This module defines a custom graph.
"""
__all__ = ["graph", "recommended_graph", "reserve_graph", "extend_graph"]


def __getattr__(name: str):
	if name == "graph":
		from src.agent.graph import graph

		return graph
	if name == "recommended_graph":
		from src.agent.recommend import recommended_graph

		return recommended_graph
	if name == "reserve_graph":
		from src.agent.reserve import reserve_graph

		return reserve_graph
	if name == "extend_graph":
		from src.agent.extend import extend_graph

		return extend_graph
	raise AttributeError(f"module 'src.agent' has no attribute {name!r}")




import unittest
import os

from langgraph.pregel import Pregel

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")

try:
    from agent.graph import graph, router_message
except ImportError:
    from src.agent.graph import graph, router_message


class TestConfiguration(unittest.TestCase):
    def test_placeholder(self) -> None:
        self.assertIsInstance(graph, Pregel)

    def test_router_message_policy(self) -> None:
        self.assertEqual(router_message({"route_destination": "POLICY_RAG"}), "policy_rag_subgraph")

    def test_router_message_recommend(self) -> None:
        self.assertEqual(router_message({"route_destination": "RECOMMEND"}), "recommend_subgraph")


if __name__ == "__main__":
    unittest.main()

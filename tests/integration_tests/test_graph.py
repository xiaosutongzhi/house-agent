import unittest
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")

try:
    from agent.graph import router_message
except ImportError:
    from src.agent.graph import router_message


class TestGraphIntegration(unittest.TestCase):
    def test_router_message_recommend(self) -> None:
        state = {"route_destination": "RECOMMEND"}
        self.assertEqual(router_message(state), "recommend_subgraph")

    def test_router_message_handoff_legacy(self) -> None:
        state = {"route": "reserve_graph"}
        self.assertEqual(router_message(state), "handoff_subgraph")


if __name__ == "__main__":
    unittest.main()

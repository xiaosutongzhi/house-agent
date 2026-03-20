import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")

try:
    import agent.node.main as main_node
except ImportError:
    import src.agent.node.main as main_node


class TestSupervisorRouter(unittest.TestCase):
    def test_supervisor_router_outcome_back_to_empathy(self) -> None:
        state = {
            "messages": [HumanMessage(content="给我推荐房源")],
            "agent_outcomes": {"recommend": {"status": "success"}},
            "route_destination": "RECOMMEND",
        }
        runtime = SimpleNamespace(context={})

        result = main_node.supervisor_router_node(state, runtime)

        self.assertEqual(result["route_destination"], "EMPATHY")
        self.assertEqual(result["route"], "empathy_generator")

    def test_supervisor_router_outcome_end_keeps_end(self) -> None:
        state = {
            "messages": [HumanMessage(content="结束")],
            "agent_outcomes": {"handoff": {"status": "done"}},
            "route_destination": "END",
        }
        runtime = SimpleNamespace(context={})

        result = main_node.supervisor_router_node(state, runtime)

        self.assertEqual(result["route_destination"], "END")
        self.assertEqual(result["route"], "empathy_generator")

    def test_supervisor_router_policy_routing_from_llm(self) -> None:
        class _DummyModel:
            @staticmethod
            def invoke(_messages):
                return AIMessage(
                    content='{"intent_level":"high","emotion_label":"neutral","route_destination":"POLICY_RAG"}'
                )

        with patch.object(main_node, "model", _DummyModel()):
            state = {"messages": [HumanMessage(content="广州首套房契税怎么交？")]}
            runtime = SimpleNamespace(context={"web_mcp_state": {"recent_action_trail": ["点击政策页"]}})
            result = main_node.supervisor_router_node(state, runtime)

        self.assertEqual(result["route_destination"], "POLICY_RAG")
        self.assertEqual(result["route"], "policy_rag_subgraph")

    def test_supervisor_router_fallback_to_keyword_recommend(self) -> None:
        class _DummyModel:
            @staticmethod
            def invoke(_messages):
                return AIMessage(content="not-a-json")

        with patch.object(main_node, "model", _DummyModel()):
            state = {"messages": [HumanMessage(content="请推荐广州3房房源")], "agent_outcomes": {}}
            runtime = SimpleNamespace(context={})
            result = main_node.supervisor_router_node(state, runtime)

        self.assertEqual(result["route_destination"], "RECOMMEND")
        self.assertEqual(result["route"], "recommend_subgraph")


if __name__ == "__main__":
    unittest.main()

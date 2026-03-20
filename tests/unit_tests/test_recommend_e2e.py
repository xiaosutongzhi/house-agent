from __future__ import annotations

import os
import unittest
import importlib
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")

recommend_node = importlib.import_module("src.agent.node.recommend")
property_store_mod = importlib.import_module("src.agent.common.property_store")
recommend_graph_mod = importlib.import_module("src.agent.recommend")
PropertyStore = property_store_mod.PropertyStore
recommended_graph = recommend_graph_mod.recommended_graph


class _FakeBoundModel:
    def __init__(self) -> None:
        self.step = 0

    def invoke(self, _messages):
        if self.step == 0:
            self.step += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "SQL_Search_Tool",
                        "args": {
                            "city": "广州",
                            "district": "天河",
                            "budget_min": 200,
                            "budget_max": 260,
                            "bedrooms": 2,
                        },
                        "id": "call_sql_1",
                        "type": "tool_call",
                    }
                ],
            )
        if self.step == 1:
            self.step += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Vector_Search_Tool",
                        "args": {
                            "query_soft_prefs": "阳台;可养宠",
                            "candidate_ids": ["GZ-TH-003"],
                            "top_k": 3,
                        },
                        "id": "call_vec_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="推荐结果：员村 南向两房 带阳台 可养宠（GZ-TH-003）")


class _FakeModel:
    def __init__(self) -> None:
        self._bound = _FakeBoundModel()

    def bind_tools(self, _tools, tool_choice=None):
        return self._bound


class RecommendE2ETests(unittest.TestCase):
    def test_recommend_subgraph_end_to_end(self):
        fake_model = _FakeModel()
        with patch.object(recommend_node, "model", fake_model):
            state = {
                "messages": [HumanMessage(content="推荐广州天河300万内两房，带阳台可养宠")],
                "user_profile": {
                    "city": "广州",
                    "district": "天河",
                    "budget_min": 200,
                    "budget_max": 260,
                    "bedrooms": 2,
                    "soft_preferences": ["阳台", "可养宠"],
                },
                "agent_outcomes": {},
            }
            result = recommended_graph.invoke(state, context={"user_id": "u-test"})

        self.assertIn("agent_outcomes", result)
        self.assertIn("recommend", result["agent_outcomes"])
        outcome = result["agent_outcomes"]["recommend"]
        self.assertEqual(outcome.get("status"), "success")
        self.assertIn("GZ-TH-003", outcome.get("final_text", ""))

        tool_names = [x.get("tool_name") for x in outcome.get("tool_observations", [])]
        self.assertIn("SQL_Search_Tool", tool_names)
        self.assertIn("Vector_Search_Tool", tool_names)

    def test_sql_tool_db_fallback(self):
        # 强制 DB 回退场景，验证 SQL 工具仍可返回候选结果
        with patch.dict(os.environ, {"DISABLE_DB": "true"}, clear=False):
            store = PropertyStore()
            self.assertFalse(store.db_enabled)

            result = recommend_node.SQL_Search_Tool.invoke(
                {
                    "city": "广州",
                    "district": "天河",
                    "budget_min": 200,
                    "budget_max": 260,
                    "bedrooms": 2,
                }
            )

        self.assertIn("candidate_ids", result)
        self.assertIn("GZ-TH-003", result["candidate_ids"])


if __name__ == "__main__":
    unittest.main()

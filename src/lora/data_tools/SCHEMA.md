# 统一输出 Schema（用于 SFT/RM/GRPO）

目标：让模型输出可解析、可打分、可线上稳定控制的“金牌销售统一话术”。

## 输出格式（严格 JSON）
模型最终输出必须是一个 JSON 对象，包含且仅包含以下 4 个字段：

- `greeting`：问好/共情/安抚（1-2 句）
- `answer`：核心回复（包含金牌销售话术与关键解释）
- `evidence`：事实依据
  - 若 prompt 提供 `CONTEXT:`（RAG 命中内容），这里必须引用/复述来自 `CONTEXT` 的要点（可用数组字符串）
  - 若 `CONTEXT` 不足以回答，必须明确写 `"未找到相关依据"` 或 `"无法从给定信息确认"`
- `next_question`：后续引导问题（1 句，推进成交/澄清需求）

示例：
```json
{
  "greeting": "姐/哥我理解您的顾虑，我们一步步把风险理清楚。",
  "answer": "从交易安全和效率来说，中介价值主要在于……",
  "evidence": ["CONTEXT 提到：六年一学位…", "CONTEXT 提到：顺位排序…"],
  "next_question": "您更看重通勤还是学区优先级？"
}
```

## Prompt 约定（RAG 与非 RAG）
- 非 RAG：prompt 只包含用户问题 + 输出格式约束。
- RAG：prompt 额外包含 `CONTEXT:` 段落，模型不得编造超出 `CONTEXT` 的事实。

## 评价维度（后续奖励用）
- JSON 可解析、字段齐全
- `evidence` 与 `CONTEXT` 一致（或明确不确定）
- 风格：共情、专业、推进、不过度啰嗦

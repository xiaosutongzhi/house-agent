#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a prompt-only RL dataset for EasyR1 GRPO.

EasyR1 examples/config.yaml expects at least:
- prompt_key: problem
- answer_key: answer

We also add optional fields:
- context: retrieval context (empty for now)
- task: coarse scenario label

You can later regenerate this file with real RAG contexts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def infer_task(instruction: str) -> str:
    for key, label in [
        ("中介费", "objection_fee"),
        ("学区", "policy_school"),
        ("贷款", "loan"),
        ("月供", "loan"),
        ("物业", "property_mgmt"),
        ("期房", "new_house"),
    ]:
        if key in instruction:
            return label
    return "general"


SCHEMA_HINT = (
    "请严格输出 JSON（不得输出多余文本），字段必须包含："
    "greeting/answer/evidence/next_question。"
    "\n- greeting：问好共情 1-2 句"
    "\n- answer：核心回复（可含金牌话术）"
    "\n- evidence：若提供 CONTEXT 必须引用/复述来自 CONTEXT；否则填空数组 []"
    "\n- next_question：1 句引导问题"
)


def build_problem(instruction: str, user_input: str, context: str) -> str:
    parts = [SCHEMA_HINT, "\n用户问题：" + instruction]
    if user_input.strip():
        parts.append("补充信息：" + user_input.strip())
    parts.append("CONTEXT:\n" + (context.strip() if context.strip() else ""))
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parents[1] / "house_agent_cleaned.jsonl"),
        help="Source jsonl with instruction/input/output.",
    )
    ap.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "easyr1" / "house_agent_grpo.jsonl"),
        help="Output jsonl for EasyR1 (will be overwritten).",
    )
    ap.add_argument("--max_samples", type=int, default=2000)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for ex in iter_jsonl(in_path):
        instruction = str(ex.get("instruction", "")).strip()
        user_input = str(ex.get("input", "")).strip()
        if not instruction:
            continue

        context = ""  # placeholder; you can fill real RAG context later
        rows.append(
            {
                "problem": build_problem(instruction, user_input, context),
                "answer": "",  # not required for RL reward; keep empty
                "context": context,
                "task": infer_task(instruction),
            }
        )
        if len(rows) >= args.max_samples:
            break

    with out_path.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} prompts to: {out_path}")


if __name__ == "__main__":
    main()

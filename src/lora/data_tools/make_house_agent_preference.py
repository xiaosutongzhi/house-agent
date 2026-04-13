#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a *pairwise preference* dataset (chosen/rejected) for RM training.

This is a pragmatic bootstrap:
- chosen: the schema JSON output from make_house_agent_schema_sft.py
- rejected: a corrupted or low-quality variant (missing field / hallucinated evidence / no next_question)

Output format matches LlamaFactory `ranking=true` + `formatting=sharegpt`:
[
  {
    "conversations": [{"from":"human","value": "..."}],
    "chosen": {"from":"gpt","value":"..."},
    "rejected": {"from":"gpt","value":"..."}
  },
  ...
]

You should later replace rejected generation with real model samples + human/LLM-judge ranking.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def corrupt_schema_json(schema_json_str: str) -> str:
    """Produce a "bad" answer variant.

    Modes:
    1) not-json
    2) json missing evidence
    3) hallucinated evidence (claiming CONTEXT without having it)
    4) remove next_question
    """

    modes = ["not_json", "missing_field", "hallucinated_evidence", "no_next_question"]
    mode = random.choice(modes)

    if mode == "not_json":
        return "当然可以。我先简单说结论：中介费就是带看房的费用。"  # deliberately wrong+unstructured

    try:
        obj = json.loads(schema_json_str)
    except Exception:
        return "我不确定。"

    if mode == "missing_field":
        obj.pop("evidence", None)
        return json.dumps(obj, ensure_ascii=False)

    if mode == "hallucinated_evidence":
        obj["evidence"] = ["根据 CONTEXT：该小区学位永久有效（示例胡编）"]
        return json.dumps(obj, ensure_ascii=False)

    if mode == "no_next_question":
        obj["next_question"] = ""
        return json.dumps(obj, ensure_ascii=False)

    return schema_json_str


def _has_context(prompt_text: str) -> bool:
    return "CONTEXT:" in prompt_text or "上下文" in prompt_text


def score_schema_answer(
    answer_text: str,
    *,
    prompt_text: str,
) -> float:
    """Heuristic rubric score for a candidate answer.

    This is intentionally biased toward *schema correctness*:
    - JSON parseability and required fields
    - evidence discipline (no hallucinated evidence when no CONTEXT)
    - non-empty next_question

    Higher is better.
    """

    # A. Format & parseability (0~4)
    try:
        obj = json.loads(answer_text)
    except Exception:
        return -10.0

    required = ["greeting", "answer", "evidence", "next_question"]
    if not all(k in obj for k in required):
        return -8.0
    if not isinstance(obj.get("greeting"), str) or not isinstance(obj.get("answer"), str) or not isinstance(
        obj.get("next_question"), str
    ):
        return -8.0
    if not isinstance(obj.get("evidence"), list) or any(not isinstance(x, str) for x in obj.get("evidence", [])):
        return -8.0

    score = 0.0
    score += 4.0

    greeting = obj.get("greeting", "").strip()
    answer = obj.get("answer", "").strip()
    evidence = obj.get("evidence", [])
    next_question = obj.get("next_question", "").strip()

    # B. Field quality (0~2)
    if greeting:
        score += 0.5
    if next_question:
        score += 1.5

    # C. Evidence discipline (0~3)
    has_ctx = _has_context(prompt_text)
    if not has_ctx:
        # Strongly prefer empty evidence when no context is provided.
        if len(evidence) == 0:
            score += 3.0
        else:
            score -= 3.0
        # Penalize claims like "根据 CONTEXT/资料/官方" without context.
        if re.search(r"根据\s*(CONTEXT|上下文|资料|政策|规定)", answer, flags=re.IGNORECASE):
            score -= 2.0
        if any(re.search(r"CONTEXT|上下文|资料|政策|规定", ev, flags=re.IGNORECASE) for ev in evidence):
            score -= 2.0
    else:
        # With context, evidence should exist but keep it small.
        if len(evidence) >= 1:
            score += 1.5
        if len(evidence) > 5:
            score -= 1.0

    # D. Answer usefulness (light heuristic, 0~1)
    if answer and len(answer) >= 20:
        score += 1.0
    return score


def generate_rejected_candidates(chosen_value: str) -> List[str]:
    """Generate a small pool of clearly-worse candidates to pick from."""

    candidates: List[str] = []
    # Ensure we always include the legacy random corruption.
    candidates.append(corrupt_schema_json(chosen_value))
    # Also include each corruption mode deterministically.
    candidates.append("当然可以。我先简单说结论：中介费就是带看房的费用。")

    try:
        obj = json.loads(chosen_value)
    except Exception:
        return candidates

    # missing_field
    obj1 = dict(obj)
    obj1.pop("evidence", None)
    candidates.append(json.dumps(obj1, ensure_ascii=False))

    # hallucinated_evidence
    obj2 = dict(obj)
    obj2["evidence"] = ["根据 CONTEXT：该小区学位永久有效（示例胡编）"]
    candidates.append(json.dumps(obj2, ensure_ascii=False))

    # no_next_question
    obj3 = dict(obj)
    obj3["next_question"] = ""
    candidates.append(json.dumps(obj3, ensure_ascii=False))

    return candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--schema_sft",
        default=str(Path(__file__).resolve().parents[1] / "LlamaFactory-main" / "data" / "house_agent_schema_sft.jsonl"),
        help="Schema SFT jsonl produced by make_house_agent_schema_sft.py",
    )
    ap.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "LlamaFactory-main" / "data" / "house_agent_pref.json"),
        help="Output preference dataset json (will be overwritten).",
    )
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--deterministic_rejected",
        action="store_true",
        help="Select rejected by heuristic rubric scoring (recommended).",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    schema_path = Path(args.schema_sft)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    for ex in iter_jsonl(schema_path):
        instruction = str(ex.get("instruction", ""))
        user_input = str(ex.get("input", ""))
        chosen_value = str(ex.get("output", "")).strip()
        if not instruction or not chosen_value:
            continue

        prompt = instruction
        if user_input:
            prompt = f"{instruction}\n\n补充信息：{user_input}"

        prompt_text_for_judge = (
            "你是房产购房顾问。请严格输出 JSON，字段包含 greeting/answer/evidence/next_question。\n\n" + prompt
        )

        if args.deterministic_rejected:
            candidates = generate_rejected_candidates(chosen_value)
            rejected_value = min(
                candidates,
                key=lambda x: score_schema_answer(x, prompt_text=prompt_text_for_judge),
            )
        else:
            rejected_value = corrupt_schema_json(chosen_value)

        samples.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt_text_for_judge,
                    }
                ],
                "chosen": {"from": "gpt", "value": chosen_value},
                "rejected": {"from": "gpt", "value": rejected_value},
            }
        )
        if len(samples) >= args.max_samples:
            break

    out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to: {out_path}")


if __name__ == "__main__":
    main()

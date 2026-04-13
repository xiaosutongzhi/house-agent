#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate schema-constrained SFT dataset for LlamaFactory.

Input (jsonl): {instruction, input, output}
Output (jsonl): same columns, but `output` is a JSON string following SCHEMA.md.

This is a *bootstrap* generator: it uses simple heuristics to fill fields.
For best quality, you can later distill/rewrite with a stronger model and replace this file.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _first_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    # Prefer first line.
    first_line = text.splitlines()[0].strip()
    if first_line:
        return first_line

    # Fallback: split by Chinese/English sentence terminators.
    m = re.split(r"(?<=[。！？!?])\s+", text, maxsplit=1)
    return (m[0] if m else text).strip()


def _infer_next_question(instruction: str) -> str:
    inst = instruction.strip()
    keyword_map = [
        ("学区", "方便说下孩子入学的时间点和年级吗？"),
        ("贷款", "您方便说下首付预算和期望贷款年限吗？"),
        ("月供", "您希望月供控制在一个大概什么区间？"),
        ("中介费", "您更在意省钱还是交易过程的省心与安全？"),
        ("物业", "您更看重安保、卫生还是报修响应速度？"),
    ]
    for kw, q in keyword_map:
        if kw in inst:
            return q
    return "方便说下您的预算、通勤和学区优先级吗？"


def build_schema_output(instruction: str, raw_output: str) -> str:
    greeting = _first_sentence(raw_output)
    next_question = ""
    # Try to reuse the last question in the raw output if exists.
    for line in reversed([ln.strip() for ln in raw_output.splitlines() if ln.strip()]):
        if "？" in line or "?" in line:
            next_question = line
            break
    if not next_question:
        next_question = _infer_next_question(instruction)

    obj = {
        "greeting": greeting,
        "answer": raw_output.strip(),
        # For non-RAG SFT bootstrap, keep evidence empty.
        "evidence": [],
        "next_question": next_question,
    }
    return json.dumps(obj, ensure_ascii=False)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parents[1] / "house_agent_cleaned.jsonl"),
        help="Input jsonl with instruction/input/output.",
    )
    ap.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "LlamaFactory-main" / "data" / "house_agent_schema_sft.jsonl"),
        help="Output jsonl path (will be overwritten).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as w:
        for ex in iter_jsonl(in_path):
            instruction = str(ex.get("instruction", ""))
            user_input = str(ex.get("input", ""))
            raw_output = str(ex.get("output", ""))
            if not instruction or not raw_output:
                continue

            ex_out = {
                "instruction": instruction,
                "input": user_input,
                "output": build_schema_output(instruction, raw_output),
            }
            w.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} examples to: {out_path}")


if __name__ == "__main__":
    main()

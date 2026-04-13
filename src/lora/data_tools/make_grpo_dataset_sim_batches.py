#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simulate batched GRPO datasets for house-agent EasyR1.

Generates:
- train batches: house_agent_grpo_sim_batch_001.jsonl ...
- validation set: house_agent_grpo_sim_val.jsonl

All rows follow fields used by EasyR1:
- problem, answer, context, task
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCHEMA_HINT = (
    "请严格输出 JSON（不得输出多余文本），字段必须包含："
    "greeting/answer/evidence/next_question。"
    "\n- greeting：问好共情 1-2 句"
    "\n- answer：核心回复（可含金牌话术）"
    "\n- evidence：若提供 CONTEXT 必须引用/复述来自 CONTEXT；否则填空数组 []"
    "\n- next_question：1 句引导问题"
)


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
        ("户型", "layout"),
        ("楼层", "layout"),
    ]:
        if key in instruction:
            return label
    return "general"


def make_context(instruction: str, task: str) -> str:
    lines = [
        f"任务类型: {task}",
        "政策提示: 购房政策可能随地区和时间调整，请以当地最新规定为准。",
        "风控提示: 不得承诺收益、不得绝对化表述。",
    ]

    if task == "loan":
        lines.append("贷款提示: 需结合首付比例、征信、收入流水综合评估。")
    elif task == "policy_school":
        lines.append("学区提示: 学位与入学资格受户籍、年限、学位占用等条件影响。")
    elif task == "objection_fee":
        lines.append("中介费提示: 服务费与服务清单应明确，避免口头承诺。")

    lines.append(f"问题摘要: {instruction[:80]}")
    return "\n".join(lines)


def build_problem(instruction: str, user_input: str, context: str) -> str:
    parts = [SCHEMA_HINT, "\n用户问题：" + instruction]
    if user_input.strip():
        parts.append("补充信息：" + user_input.strip())
    parts.append("CONTEXT:\n" + context)
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parents[1] / "house_agent_cleaned.jsonl"),
        help="Source jsonl with instruction/input/output.",
    )
    ap.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parents[1] / "easyr1" / "sim_batches"),
        help="Output directory for batched simulated GRPO datasets.",
    )
    ap.add_argument("--max_samples", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--val_size", type=int, default=80)
    ap.add_argument("--with_context_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for ex in iter_jsonl(in_path):
        instruction = str(ex.get("instruction", "")).strip()
        user_input = str(ex.get("input", "")).strip()
        if not instruction:
            continue

        task = infer_task(instruction)
        context = make_context(instruction, task) if rng.random() < args.with_context_ratio else ""
        rows.append(
            {
                "problem": build_problem(instruction, user_input, context),
                "answer": "",
                "context": context,
                "task": task,
            }
        )
        if len(rows) >= args.max_samples:
            break

    rng.shuffle(rows)
    val_size = min(max(args.val_size, 1), max(len(rows) - 1, 1))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    val_file = out_dir / "house_agent_grpo_sim_val.jsonl"
    with val_file.open("w", encoding="utf-8") as w:
        for r in val_rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_batches = 0
    for i in range(0, len(train_rows), args.batch_size):
        batch = train_rows[i : i + args.batch_size]
        if not batch:
            continue
        n_batches += 1
        batch_file = out_dir / f"house_agent_grpo_sim_batch_{n_batches:03d}.jsonl"
        with batch_file.open("w", encoding="utf-8") as w:
            for r in batch:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "input": str(in_path),
        "out_dir": str(out_dir),
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "val_size": val_size,
        "train_size": len(train_rows),
        "n_batches": n_batches,
        "with_context_ratio": args.with_context_ratio,
        "seed": args.seed,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as w:
        json.dump(meta, w, ensure_ascii=False, indent=2)

    print(f"[sim-grpo] train_size={len(train_rows)} val_size={len(val_rows)} batches={n_batches}")
    print(f"[sim-grpo] val_file={val_file}")


if __name__ == "__main__":
    main()

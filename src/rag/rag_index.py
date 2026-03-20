from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.rag.retrieval import extract_cities, extract_tax_types, parse_years, tokenize


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？；;])\s*|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def strip_front_matter(lines: List[str]) -> List[str]:
    if not lines or lines[0].strip() != "---":
        return lines
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[i + 1 :]
    return lines


def iter_sentences(md_text: str, source: str) -> Iterable[Dict[str, Any]]:
    lines = strip_front_matter(md_text.splitlines())
    section = None
    sent_id = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("##"):
            section = line.lstrip("#").strip()
            continue
        for sent in split_sentences(line):
            meta = {
                "source": source,
                "section": section,
            }
            meta["city"] = extract_cities(sent)
            meta["tax_type"] = extract_tax_types(sent)
            meta["years"] = parse_years(sent)
            yield {
                "sent_id": sent_id,
                "text": sent,
                "tokens": tokenize(sent),
                "meta": meta,
            }
            sent_id += 1


def build_index(
    input_md: str,
    output_jsonl: str,
    use_embeddings: bool = False,
    embedding_model: Optional[str] = None,
) -> None:
    with open(input_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    items = list(iter_sentences(md_text, os.path.basename(input_md)))
    embeddings = None

    if use_embeddings:
        try:
            import importlib

            OpenAIEmbeddings = importlib.import_module("langchain_openai").OpenAIEmbeddings
        except Exception:
            raise RuntimeError("langchain_openai 未安装，无法生成向量嵌入")

        model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        embeddings = OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url,chunk_size=64)
        texts = [item["text"] for item in items]
        vectors = embeddings.embed_documents(texts)
        for item, vec in zip(items, vectors):
            item["embedding"] = vec

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in items:
            row = {
                "doc_id": os.path.basename(input_md),
                "sent_id": item["sent_id"],
                "text": item["text"],
                "tokens": item["tokens"],
                "meta": item["meta"],
                "embedding": item.get("embedding"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ index built: {output_jsonl} (items={len(items)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建政策 RAG 索引")
    parser.add_argument("--input", required=True, help="清洗后的 Markdown 文件")
    parser.add_argument("--output", required=True, help="输出 JSONL 索引路径")
    parser.add_argument("--use-embeddings", action="store_true", help="启用向量嵌入")
    parser.add_argument("--embedding-model", default=None, help="Embedding 模型名")
    return parser.parse_args()


def main() -> None:
    try:
        import importlib

        load_dotenv = importlib.import_module("dotenv").load_dotenv
        load_dotenv()
    except Exception:
        pass
    if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE", "")
    args = parse_args()
    build_index(
        input_md=args.input,
        output_jsonl=args.output,
        use_embeddings=args.use_embeddings,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

CITIES = [
    "北京",
    "上海",
    "广州",
    "深圳",
    "天津",
    "重庆",
    "杭州",
    "南京",
    "苏州",
    "成都",
    "武汉",
    "西安",
]

TAX_TYPES = {
    "契税": "契税",
    "增值税": "增值税",
    "个人所得税": "个税",
    "个税": "个税",
    "土地增值税": "土地增值税",
    "房产税": "房产税",
    "城镇土地使用税": "城镇土地使用税",
}


@dataclass
class IndexItem:
    doc_id: str
    sent_id: int
    text: str
    tokens: List[str]
    meta: Dict[str, Any]
    embedding: Optional[List[float]]


class BM25Index:
    def __init__(self, items: List[IndexItem], k1: float = 1.5, b: float = 0.75) -> None:
        self.items = items
        self.k1 = k1
        self.b = b
        self.doc_len = [len(item.tokens) for item in items]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.df: Dict[str, int] = {}
        for item in items:
            for term in set(item.tokens):
                self.df[term] = self.df.get(term, 0) + 1
        self.N = len(items)
        self.idf: Dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: List[str], idx: int) -> float:
        if not query_tokens:
            return 0.0
        item = self.items[idx]
        freqs: Dict[str, int] = {}
        for t in item.tokens:
            freqs[t] = freqs.get(t, 0) + 1
        score = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            idf = self.idf.get(term, 0.0)
            tf = freqs[term]
            denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / self.avgdl))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score


def tokenize(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text.lower())


def extract_cities(text: str) -> List[str]:
    found = [city for city in CITIES if city in text]
    if "北上广深" in text:
        found.extend(["北京", "上海", "广州", "深圳"])
    if "一线城市" in text:
        found.append("一线城市")
    return sorted(set(found))


def extract_tax_types(text: str) -> List[str]:
    found = []
    for k, v in TAX_TYPES.items():
        if k in text:
            found.append(v)
    return sorted(set(found))


def parse_years(text: str) -> List[int]:
    years = re.findall(r"(20\d{2})", text)
    return sorted({int(y) for y in years})


def load_index(path: str) -> List[IndexItem]:
    items: List[IndexItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                IndexItem(
                    doc_id=row["doc_id"],
                    sent_id=row["sent_id"],
                    text=row["text"],
                    tokens=row.get("tokens") or [],
                    meta=row.get("meta") or {},
                    embedding=row.get("embedding"),
                )
            )
    return items


def build_doc_map(items: List[IndexItem]) -> Dict[str, List[IndexItem]]:
    doc_map: Dict[str, List[IndexItem]] = {}
    for item in items:
        doc_map.setdefault(item.doc_id, []).append(item)
    for doc_id in doc_map:
        doc_map[doc_id].sort(key=lambda x: x.sent_id)
    return doc_map


def match_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    if not filters:
        return True
    city = filters.get("city")
    if city:
        meta_cities = meta.get("city") or []
        if city not in meta_cities:
            return False
    tax_type = filters.get("tax_type")
    if tax_type:
        meta_types = meta.get("tax_type") or []
        if tax_type not in meta_types:
            return False
    return True


def bm25_search(
    items: List[IndexItem],
    bm25: BM25Index,
    query: str,
    filters: Optional[Dict[str, Any]],
    top_k: int = 8,
) -> List[Tuple[int, float]]:
    query_tokens = tokenize(query)
    scored = []
    for idx, item in enumerate(items):
        if filters and not match_filters(item.meta, filters):
            continue
        score = bm25.score(query_tokens, idx)
        if score > 0:
            scored.append((idx, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def dense_search(
    items: List[IndexItem],
    query_embedding: List[float],
    filters: Optional[Dict[str, Any]],
    top_k: int = 8,
) -> List[Tuple[int, float]]:
    if not query_embedding:
        return []
    q = np.array(query_embedding, dtype=np.float32)
    scored = []
    for idx, item in enumerate(items):
        if not item.embedding:
            continue
        if filters and not match_filters(item.meta, filters):
            continue
        v = np.array(item.embedding, dtype=np.float32)
        denom = (np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
        sim = float(np.dot(q, v) / denom)
        scored.append((idx, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def rrf_fuse(rankings: List[List[int]], k: int = 60) -> List[int]:
    scores: Dict[int, float] = {}
    for rank_list in rankings:
        for rank, idx in enumerate(rank_list):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def build_windows(
    doc_map: Dict[str, List[IndexItem]],
    items: List[IndexItem],
    hit_indices: List[int],
    window: int = 2,
) -> List[str]:
    contexts = []
    for idx in hit_indices:
        item = items[idx]
        doc_items = doc_map[item.doc_id]
        sent_ids = [i.sent_id for i in doc_items]
        pos = sent_ids.index(item.sent_id)
        start = max(pos - window, 0)
        end = min(pos + window + 1, len(doc_items))
        window_text = "".join([x.text for x in doc_items[start:end]])
        contexts.append(window_text)
    return contexts


def get_default_index_path() -> str:
    return os.getenv(
        "POLICY_INDEX_PATH",
        os.path.join(os.path.dirname(__file__), "policy_index.jsonl"),
    )

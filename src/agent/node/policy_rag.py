from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from src.agent.common.context import ContextSchema
from src.agent.common.llm import model
from src.agent.state.main import State
from src.rag.retrieval import (
    BM25Index,
    build_doc_map,
    bm25_search,
    dense_search,
    get_default_index_path,
    load_index,
    rrf_fuse,
)


class PolicyQuery(BaseModel):
    rewritten_query: str = Field(description="结合记忆与上下文改写后的检索查询")
    city: Optional[str] = Field(default=None, description="适用城市，如上海、北京")
    tax_type: Optional[str] = Field(default=None, description="税种，如契税、增值税、个税")
    area_sqm: Optional[int] = Field(default=None, description="面积（平方米）")
    house_count: Optional[int] = Field(default=None, description="名下住房套数")


class PolicyProfile(BaseModel):
    city: Optional[str] = None
    house_count: Optional[int] = None
    area_sqm: Optional[int] = None


_INDEX_CACHE = {
    "items": None,
    "bm25": None,
    "doc_map": None,
}


def _load_index() -> None:
    if _INDEX_CACHE["items"] is not None:
        return
    index_path = get_default_index_path()
    items = load_index(index_path)
    _INDEX_CACHE["items"] = items
    _INDEX_CACHE["bm25"] = BM25Index(items)
    _INDEX_CACHE["doc_map"] = build_doc_map(items)


def _get_profile(runtime: Runtime[ContextSchema], store: BaseStore | None) -> PolicyProfile:
    if store is None:
        return PolicyProfile()

    user_id = runtime.context.get("user_id")
    namespace = (user_id, "policy_profile")
    results = store.search(namespace)
    if results and results[0]:
        return PolicyProfile(**results[0].value)
    return PolicyProfile()


def _update_profile(runtime: Runtime[ContextSchema], store: BaseStore | None, update: PolicyProfile) -> None:
    if store is None:
        return

    user_id = runtime.context.get("user_id")
    namespace = (user_id, "policy_profile")
    results = store.search(namespace)
    data = update.model_dump(exclude_none=True)
    if not data:
        return
    if results and results[0]:
        existing = results[0].value
        existing.update(data)
        store.put(namespace, results[0].key, existing)
    else:
        store.put(namespace, "profile", data)


def _rewrite_query(question: str, profile: PolicyProfile) -> PolicyQuery:
    system_prompt = (
        "你是房地产政策检索改写专家。需要结合用户长期记忆信息，将用户问题改写为可检索的完整查询。"
        "只输出结构化结果，不要编造。"
    )
    user_prompt = (
        f"用户长期记忆：城市={profile.city}，名下套数={profile.house_count}，面积={profile.area_sqm}平方米。\n"
        f"用户问题：{question}\n"
        "请输出改写后的查询，并提取城市与税种。"
    )
    json_prompt = (
        "请仅输出JSON对象，不要输出其它文本。"
        "字段：rewritten_query(必填), city, tax_type, area_sqm, house_count。"
    )
    raw = model.invoke(
        [SystemMessage(content=system_prompt + "\n" + json_prompt), HumanMessage(content=user_prompt)]
    )
    text = getattr(raw, "content", "") or ""

    raw_obj: dict = {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            raw_obj = obj
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    raw_obj = obj
            except Exception:
                raw_obj = {}

    rewritten_query = (
        raw_obj.get("rewritten_query")
        or raw_obj.get("query")
        or raw_obj.get("rewrite")
        or question
    )

    def _to_int(v):
        if v is None or v == "":
            return None
        try:
            return int(str(v).strip())
        except Exception:
            return None

    return PolicyQuery(
        rewritten_query=str(rewritten_query),
        city=raw_obj.get("city"),
        tax_type=raw_obj.get("tax_type") or raw_obj.get("tax"),
        area_sqm=_to_int(raw_obj.get("area_sqm") or raw_obj.get("area")),
        house_count=_to_int(raw_obj.get("house_count") or raw_obj.get("count")),
    )


def _build_contexts(query: PolicyQuery) -> List[str]:
    _load_index()
    items = _INDEX_CACHE["items"]
    bm25 = _INDEX_CACHE["bm25"]
    doc_map = _INDEX_CACHE["doc_map"]

    filters = {}
    if query.city:
        filters["city"] = query.city
    if query.tax_type:
        filters["tax_type"] = query.tax_type

    bm25_hits = bm25_search(items, bm25, query.rewritten_query, filters=filters, top_k=8)
    bm25_rank = [idx for idx, _ in bm25_hits]

    dense_rank: List[int] = []
    if items and items[0].embedding is not None:
        try:
            import importlib

            OpenAIEmbeddings = importlib.import_module("langchain_openai").OpenAIEmbeddings
            emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            embed = OpenAIEmbeddings(model=emb_model)
            q_emb = embed.embed_query(query.rewritten_query)
            dense_hits = dense_search(items, q_emb, filters=filters, top_k=8)
            dense_rank = [idx for idx, _ in dense_hits]
        except Exception:
            dense_rank = []

    merged = rrf_fuse([bm25_rank, dense_rank] if dense_rank else [bm25_rank])
    top_indices = merged[:5]

    contexts: List[str] = []
    for idx in top_indices:
        item = items[idx]
        doc_items = doc_map[item.doc_id]
        sent_ids = [x.sent_id for x in doc_items]
        pos = sent_ids.index(item.sent_id)
        start = max(pos - 2, 0)
        end = min(pos + 3, len(doc_items))
        window_text = "".join([x.text for x in doc_items[start:end]])
        contexts.append(window_text)
    return contexts


def policy_rag_node(state: State, runtime: Runtime[ContextSchema], *, store: BaseStore | None = None):
    question = state["messages"][-1].content
    profile = _get_profile(runtime, store)
    query = _rewrite_query(question, profile)

    # 更新长期记忆
    _update_profile(
        runtime,
        store,
        PolicyProfile(city=query.city, house_count=query.house_count, area_sqm=query.area_sqm),
    )

    contexts = _build_contexts(query)
    context_text = "\n\n".join(contexts) if contexts else ""

    system_prompt = (
        "你是房地产政策问答助手。仅基于给定上下文回答问题；如果上下文没有信息，请明确说明未找到。"
    )
    user_prompt = (
        f"问题：{question}\n\n"
        f"检索上下文：\n{context_text}\n\n"
        "请给出简洁、结构化的回答。"
    )

    answer = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return {"messages": [answer]}


def policy_rewrite_node(state: State, runtime: Runtime[ContextSchema], *, store: BaseStore | None = None):
    question = state["messages"][-1].content
    profile = _get_profile(runtime, store)
    query = _rewrite_query(question, profile)
    mcp_context = runtime.context.get("mcp_context") or runtime.context.get("web_mcp_state") or {}
    if isinstance(mcp_context, dict) and mcp_context.get("focus_property"):
        query.rewritten_query = f"{query.rewritten_query}（关注楼盘:{mcp_context.get('focus_property')}）"

    _update_profile(
        runtime,
        store,
        PolicyProfile(city=query.city, house_count=query.house_count, area_sqm=query.area_sqm),
    )
    return {"policy_query": query.model_dump()}


def policy_retrieve_node(state: State):
    query_data = state.get("policy_query") or {}
    query = PolicyQuery(**query_data)
    contexts = _build_contexts(query)
    return {"policy_contexts": contexts}


def policy_rerank_node(state: State):
    contexts = state.get("policy_contexts") or []
    if not contexts:
        return {"policy_ranked_contexts": []}
    # 轻量精排：按长度优先（示例占位，可替换 cross-encoder）
    ranked = sorted(contexts, key=lambda x: len(x), reverse=True)[:5]
    return {"policy_ranked_contexts": ranked}


def policy_generate_node(state: State):
    question = state["messages"][-1].content
    contexts = state.get("policy_ranked_contexts") or []
    context_text = "\n\n".join(contexts) if contexts else ""

    system_prompt = "你是房地产政策问答助手。仅基于给定上下文回答；若无依据请明确说明。"
    user_prompt = (
        f"问题：{question}\n\n"
        f"检索上下文：\n{context_text}\n\n"
        "请给出结构化答案（结论+依据+注意事项）。"
    )
    answer = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return {
        "messages": [answer],
        "agent_outcomes": {
            **(state.get("agent_outcomes") or {}),
            "policy_rag": {
                "status": "ok" if contexts else "not_found",
                "contexts": contexts,
                "answer": getattr(answer, "content", str(answer)),
            },
        },
    }

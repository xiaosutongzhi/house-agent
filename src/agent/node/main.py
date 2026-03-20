import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, filter_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.agent.common.context import ContextSchema
from src.agent.common.llm import model
from src.agent.state.main import NeedReserveOutput, State


# 节点：查询持久化信息
def get_store_info(state: State, runtime: Runtime[ContextSchema], * , store: BaseStore | None = None):
    # 搜索用户信息
    if store is None:
        return {"user_preferences": {}}

    user_id = runtime.context.get("user_id")
    namespace = (user_id, "preferences")
    prefs_result = store.search(namespace)
    if prefs_result and prefs_result[0]:
        return {
            "user_preferences": prefs_result[0].value
        }
    else:
        return {
            "user_preferences": {}
        }


class LongTermMemory(BaseModel):
    preferred_city: Optional[str] = Field(default=None, description="长期偏好城市")
    preferred_district: Optional[str] = Field(default=None, description="长期偏好区域")
    budget_min: Optional[float] = Field(default=None, description="长期最低预算")
    budget_max: Optional[float] = Field(default=None, description="长期最高预算")
    room_type: Optional[str] = Field(default=None, description="偏好房型")
    orientation: Optional[str] = Field(default=None, description="偏好朝向")
    others: Optional[str] = Field(default=None, description="其它长期偏好")
    tags: Optional[list[str]] = Field(default=None, description="偏好标签列表")


class PreferenceExtraction(BaseModel):
    structured_data: LongTermMemory = Field(
        default_factory=LongTermMemory,
        description="明确、可覆盖更新的硬性指标（结构化画像）",
    )
    semantic_data: list[str] = Field(
        default_factory=list,
        description="软性偏好/关注点碎片（用于向量记忆追加），尽量用短语列表表达",
    )


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _normalize_memory_update(raw_obj: dict[str, Any]) -> dict[str, Any]:
    # 字段别名兼容
    city = raw_obj.get("preferred_city") or raw_obj.get("city")
    district = raw_obj.get("preferred_district") or raw_obj.get("district")
    budget_min = _safe_float(raw_obj.get("budget_min") or raw_obj.get("min_budget"))
    budget_max = _safe_float(raw_obj.get("budget_max") or raw_obj.get("max_budget"))
    room_type = raw_obj.get("room_type")
    orientation = raw_obj.get("orientation")
    others = raw_obj.get("others")
    tags = raw_obj.get("tags")

    if isinstance(tags, str):
        tags = [x.strip() for x in re.split(r"[,，、\s]+", tags) if x.strip()]
    if tags is not None and not isinstance(tags, list):
        tags = None

    data = {
        "preferred_city": city,
        "preferred_district": district,
        "budget_min": budget_min,
        "budget_max": budget_max,
        "room_type": room_type,
        "orientation": orientation,
        "others": others,
        "tags": tags,
    }
    return {k: v for k, v in data.items() if v is not None}


def _try_import_tiktoken():
    try:
        import tiktoken  # type: ignore

        return tiktoken
    except Exception:
        return None


def _get_encoding_for_model(model_name: str | None):
    tiktoken = _try_import_tiktoken()
    if tiktoken is None:
        return None

    name = (model_name or "").strip()
    try:
        return tiktoken.encoding_for_model(name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _message_text(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    # 兼容多模态/结构化 content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _count_tokens(messages: list[Any], *, model_name: str | None = None) -> int:
    """使用 tiktoken 对消息列表做近似 token 统计。

    若 tiktoken 不可用，则退化为字符级粗略估计（不会中断图执行）。
    """
    encoding = _get_encoding_for_model(model_name)
    total = 0
    for msg in messages:
        text = _message_text(msg)
        if not text:
            continue
        if encoding is None:
            # 经验系数：中文/英文混合下按 1 token ~= 4 chars 粗估
            total += max(1, len(text) // 4)
        else:
            total += len(encoding.encode(text))
    return total


def _format_for_summary(messages: list[Any]) -> str:
    lines: list[str] = []
    for m in messages:
        role = getattr(m, "type", None) or m.__class__.__name__
        role = str(role)
        text = _message_text(m).strip()
        if not text:
            continue
        lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def _llm_summarize_old_context(old_messages: list[Any]) -> str:
    """用轻量 prompt 总结旧上下文（失败则返回空字符串）。"""
    if not old_messages:
        return ""

    prompt = (
        "请提取并总结以下对话中的购房线索与核心进展，忽略寒暄。\n"
        "要求：\n"
        "- 只保留事实与条件变化（预算/区域/户型/取舍/决策进度）\n"
        "- 用要点列表输出（不超过 12 条）\n"
        "- 不要编造不存在的信息\n\n"
        "对话：\n"
        f"{_format_for_summary(old_messages)}"
    )
    msgs = [
        SystemMessage(content="你是对话摘要助手。"),
        HumanMessage(content=prompt),
    ]

    try:
        res = model.invoke(msgs)
        return (getattr(res, "content", "") or "").strip()
    except Exception:
        return ""


def _maybe_compress_messages(
    messages: list[Any],
    *,
    max_tokens: int = 3000,
    keep_last_turns: int = 3,
) -> list[Any]:
    """长上下文滑动窗口截断：保留系统提示词 + 最近 N 轮对话，其余压缩为摘要。"""
    if not messages:
        return messages

    # 先判断是否超阈值
    model_name = None
    try:
        model_name = getattr(model, "model", None) or getattr(model, "model_name", None)
    except Exception:
        model_name = None
    total_tokens = _count_tokens(messages, model_name=str(model_name) if model_name else None)
    if total_tokens <= max_tokens:
        return messages

    # 1) 保留开头的 system messages
    sys_end = 0
    for m in messages:
        if isinstance(m, SystemMessage):
            sys_end += 1
        else:
            break

    # 2) 保留最近 N 轮（粗略按 2 条消息一轮：human+ai）
    keep_last_messages = max(2, keep_last_turns * 2)
    cut_index = max(sys_end, len(messages) - keep_last_messages)
    if cut_index <= sys_end:
        return messages

    old = messages[sys_end:cut_index]
    recent = messages[cut_index:]
    summary = _llm_summarize_old_context(old)
    if not summary:
        # 总结失败：退化为硬截断（仍尽量保留 system + 最近对话）
        return messages[:sys_end] + recent

    summary_msg = SystemMessage(
        content=(
            "以下是之前对话的摘要（用于续聊，不要逐字复述）：\n" + summary
        )
    )
    compressed = messages[:sys_end] + [summary_msg] + recent

    # 二次保障：若仍超阈值，直接增加截断力度
    if _count_tokens(compressed, model_name=str(model_name) if model_name else None) > max_tokens:
        hard_recent = recent[-keep_last_messages:]
        compressed = messages[:sys_end] + [summary_msg] + hard_recent
    return compressed


def _project_root() -> Path:
    # .../src/agent/node/main.py -> parents: node/agent/src/house-agent
    return Path(__file__).resolve().parents[3]


def _default_sqlite_path() -> Path:
    return _project_root() / "data" / "memory.sqlite"


def _sqlite_upsert_structured_profile(user_id: str, data: dict[str, Any]) -> None:
    """轨道A：结构化画像 Upsert 到 SQLite（字段覆盖更新）。"""
    if not user_id:
        return
    db_path = Path(str(os.getenv("MEMORY_SQLITE_PATH"))) if os.getenv("MEMORY_SQLITE_PATH") else _default_sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    import sqlite3

    con = sqlite3.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profile (
              user_id TEXT PRIMARY KEY,
              data_json TEXT NOT NULL,
              updated_at INTEGER NOT NULL
            )
            """
        )
        now = int(time.time())
        con.execute(
            """
            INSERT INTO user_profile(user_id, data_json, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              data_json=excluded.data_json,
              updated_at=excluded.updated_at
            """,
            (user_id, json.dumps(data, ensure_ascii=False), now),
        )
        con.commit()
    finally:
        con.close()


def _sqlite_append_semantic_fragment(user_id: str, text: str, embedding: list[float] | None) -> None:
    """轨道B：语义碎片追加到 SQLite（作为无 Chroma 时的向量库兜底）。"""
    if not user_id or not text:
        return
    db_path = Path(str(os.getenv("MEMORY_SQLITE_PATH"))) if os.getenv("MEMORY_SQLITE_PATH") else _default_sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    import sqlite3

    con = sqlite3.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_memory (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              text TEXT NOT NULL,
              embedding_json TEXT
            )
            """
        )
        now = int(time.time())
        con.execute(
            """
            INSERT INTO semantic_memory(user_id, created_at, text, embedding_json)
            VALUES(?, ?, ?, ?)
            """,
            (user_id, now, text, json.dumps(embedding) if embedding else None),
        )
        con.commit()
    finally:
        con.close()


def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    """尽力向量化：优先 OpenAIEmbeddings，否则返回 None。"""
    if not texts:
        return None
    try:
        from langchain_openai import OpenAIEmbeddings

        emb_model = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        embeddings = OpenAIEmbeddings(model=emb_model)
        return embeddings.embed_documents(texts)
    except Exception:
        return None


def _try_chroma_add(user_id: str, texts: list[str], embeddings: list[list[float]] | None) -> bool:
    """可选：写入 Chroma（如果安装且配置了持久化目录）。失败则返回 False。"""
    try:
        import chromadb  # type: ignore

        persist_dir = os.getenv("CHROMA_PERSIST_DIR") or str(_project_root() / "data" / "chroma")
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(name="semantic_memory")
        now = int(time.time())
        ids = [f"{user_id}:{now}:{i}" for i in range(len(texts))]
        metadatas = [{"user_id": user_id, "created_at": now} for _ in texts]
        if embeddings is not None:
            collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        else:
            collection.add(ids=ids, documents=texts, metadatas=metadatas)
        return True
    except Exception:
        return False


def memory_manager_node(state: State, runtime: Runtime[ContextSchema], *, store: BaseStore | None = None):
    """短期滑窗压缩 + 长期双轨记忆管理。

    Step1: Token 计数 + 滑动窗口压缩（替换旧消息为摘要）。
    Step2: 从最新输入提取 structured_data / semantic_data。
    Step3: structured_data 做 Upsert 覆盖；semantic_data 做向量化追加。
    """
    mcp_context = runtime.context.get("mcp_context") or runtime.context.get("web_mcp_state") or {}

    user_id = runtime.context.get("user_id")
    if not user_id:
        user_id = "anonymous"

    if store is None:
        user_preferences = {}
        mem_namespace = (user_id, "long_term_memory")
        mem_result = []
        current_memory = {}
    else:
        prefs_namespace = (user_id, "preferences")
        prefs_result = store.search(prefs_namespace)
        user_preferences = prefs_result[0].value if prefs_result and prefs_result[0] else {}

        mem_namespace = (user_id, "long_term_memory")
        mem_result = store.search(mem_namespace)
        current_memory = mem_result[0].value if mem_result and mem_result[0] else {}

    # ---------- Step1: 短期记忆（滑窗压缩） ----------
    compressed_messages = _maybe_compress_messages(
        list(state.get("messages") or []),
        max_tokens=int(os.getenv("MAX_TOKENS", "3000")),
        keep_last_turns=int(os.getenv("KEEP_LAST_TURNS", "3")),
    )

    # ---------- Step2: 偏好抽取（structured + semantic） ----------
    user_messages = filter_messages(compressed_messages, include_types="human")
    # 读取最新 1-2 轮用户输入，避免只用最后一句丢信息
    last_user_texts = [m.content for m in user_messages[-2:]] if user_messages else []
    last_user_msg = "\n".join([t for t in last_user_texts if isinstance(t, str)])
    if mcp_context:
        last_user_msg += f"\n隐式上下文: {json.dumps(mcp_context, ensure_ascii=False)}"

    extract_system = (
        "你是购房偏好信息抽取器。请从用户的最新输入中抽取偏好，并按结构化输出返回。\n"
        "约束：只抽取用户明确表达的条件，不要猜测。\n"
        "structured_data：硬性条件（预算/区域/房型/朝向等），用于覆盖更新。\n"
        "semantic_data：软性偏好短语列表（如‘喜欢大阳台’‘看重绿化’），用于追加到语义记忆。"
    )
    extract_user = (
        f"当前结构化画像（可能为空）：{json.dumps(current_memory, ensure_ascii=False)}\n"
        f"用户最新输入：{last_user_msg}\n"
        "请输出 structured_data 与 semantic_data。"
    )

    extraction: PreferenceExtraction | None = None
    try:
        extractor = model.with_structured_output(PreferenceExtraction)
        extraction = extractor.invoke(
            [SystemMessage(content=extract_system), HumanMessage(content=extract_user)]
        )
    except Exception:
        extraction = None

    # 结构化输出兜底：JSON
    if extraction is None:
        json_prompt = (
            "请仅输出JSON对象，不要输出其它文本。必须包含 structured_data 与 semantic_data 两个字段。"
            "structured_data 可包含：preferred_city, preferred_district, budget_min, budget_max, room_type, orientation, others, tags。"
            "semantic_data 为字符串数组。"
        )
        try:
            raw = model.invoke(
                [
                    SystemMessage(content=extract_system + "\n" + json_prompt),
                    HumanMessage(content=extract_user),
                ]
            )
            raw_obj = _extract_json_object(getattr(raw, "content", ""))
            structured_raw = raw_obj.get("structured_data") or {}
            semantic_raw = raw_obj.get("semantic_data") or []
            if isinstance(semantic_raw, str):
                semantic_raw = [semantic_raw]
            if not isinstance(semantic_raw, list):
                semantic_raw = []
            extraction = PreferenceExtraction(
                structured_data=LongTermMemory(**structured_raw) if isinstance(structured_raw, dict) else LongTermMemory(),
                semantic_data=[str(x).strip() for x in semantic_raw if str(x).strip()],
            )
        except Exception:
            extraction = PreferenceExtraction()

    structured_update = _normalize_memory_update(extraction.structured_data.model_dump(exclude_none=True))
    semantic_fragments = [s for s in (extraction.semantic_data or []) if s and s.strip()]

    # ---------- Step3: 双轨记忆持久化 ----------
    # 轨道A（核心画像 - Upsert/Overwrite）：只覆盖本轮抽取到的字段
    merged = dict(current_memory)
    if structured_update:
        merged.update(structured_update)

    # A1) 写入 LangGraph Store（如果存在）
    if store is not None:
        if mem_result and mem_result[0]:
            store.put(mem_namespace, mem_result[0].key, merged)
        else:
            store.put(mem_namespace, "profile", merged)

    # A2) 写入 SQLite（关系型 Upsert）
    try:
        _sqlite_upsert_structured_profile(user_id, merged)
    except Exception:
        # 不因持久化失败阻断主流程
        pass

    # 轨道B（情景碎片 - Append with Timestamp）：向量化后追加
    if semantic_fragments:
        # 伪代码（真实执行见下方 best-effort 实现）：
        # embeddings = embed(semantic_fragments)
        # vector_db.upsert_or_add(user_id_namespace, text=fragment, embedding=emb, ts=now)
        try:
            embeddings = _embed_texts(semantic_fragments)
            wrote = _try_chroma_add(user_id, semantic_fragments, embeddings)
            if not wrote:
                for i, frag in enumerate(semantic_fragments):
                    emb = embeddings[i] if embeddings and i < len(embeddings) else None
                    _sqlite_append_semantic_fragment(user_id, frag, emb)
        except Exception:
            # 追加失败：忽略
            pass

    # 映射到新架构 user_profile
    soft_from_memory = merged.get("tags") or []
    if isinstance(soft_from_memory, str):
        soft_from_memory = [soft_from_memory]
    implicit_focus = mcp_context.get("focus_property") if isinstance(mcp_context, dict) else None
    if implicit_focus:
        soft_from_memory = list(dict.fromkeys([*soft_from_memory, f"关注楼盘:{implicit_focus}"]))

    user_profile = {
        "budget_min": merged.get("budget_min"),
        "budget_max": merged.get("budget_max"),
        "city": merged.get("preferred_city"),
        "district": merged.get("preferred_district"),
        "room_type": merged.get("room_type"),
        "orientation": merged.get("orientation"),
        "soft_preferences": [x for x in soft_from_memory if x],
        "emotion": state.get("emotion_label", "neutral"),
    }

    return {
        "messages": compressed_messages,
        "user_preferences": user_preferences,
        "user_memory": merged,
        "user_profile": {k: v for k, v in user_profile.items() if v is not None},
    }


class EmotionRouteDecision(BaseModel):
    intent_level: Literal["low", "medium", "high"] = Field(description="购买意向等级")
    emotion_label: Literal["positive", "neutral", "negative", "anxious", "urgent"] = Field(
        description="情绪标签"
    )
    route_destination: Literal["RECOMMEND", "POLICY_RAG", "HANDOFF", "EMPATHY", "END"] = Field(
        description="下游路由"
    )


def _normalize_intent_level(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in {"high", "h", "高", "高意向"}:
        return "high"
    if v in {"low", "l", "低", "低意向"}:
        return "low"
    return "medium"


def _normalize_emotion_label(value: str | None) -> str:
    v = (value or "").strip().lower()
    if any(k in v for k in ["焦虑", "anxious", "担心", "紧张"]):
        return "anxious"
    if any(k in v for k in ["急", "紧急", "urgent", "马上"]):
        return "urgent"
    if any(k in v for k in ["负面", "negative", "不满", "生气"]):
        return "negative"
    if any(k in v for k in ["积极", "positive", "开心", "满意"]):
        return "positive"
    return "neutral"


def _normalize_route(value: str | None, user_text: str) -> str:
    v = (value or "").strip().lower()
    t = (user_text or "").lower()

    if any(k in v for k in ["policy", "政策", "税", "契税", "增值税", "个税"]):
        return "POLICY_RAG"
    if any(k in v for k in ["reserve", "预订", "下单", "工单", "预约", "人工"]):
        return "HANDOFF"
    if any(k in v for k in ["recommend", "找房", "推荐", "房源", "看房"]):
        return "RECOMMEND"
    if any(k in v for k in ["empathy", "总结", "出口"]):
        return "EMPATHY"
    if any(k in v for k in ["end", "结束"]):
        return "END"

    # 兜底：根据用户原始问题判断
    if any(k in t for k in ["政策", "税", "契税", "增值税", "个税", "优惠"]):
        return "POLICY_RAG"
    if any(k in t for k in ["预订", "下单", "预约", "工单", "人工"]):
        return "HANDOFF"
    if any(k in t for k in ["推荐", "找房", "房源", "看房"]):
        return "RECOMMEND"
    return "HANDOFF"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    # 优先直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 再尝试提取首个 JSON 对象
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def supervisor_router_node(state: State, runtime: Runtime[ContextSchema]):
    """中心枢纽路由节点：仅做确定性分发，不做工具推理。"""
    # 子图回传后统一汇聚：由 Supervisor 决定下一跳（子图不改路由）
    outcomes = state.get("agent_outcomes") or {}
    if any(k in outcomes for k in ("recommend", "policy_rag", "handoff")):
        next_dest = "END" if state.get("route_destination") == "END" else "EMPATHY"
        return {
            "route_destination": next_dest,
            "route": "empathy_generator",
        }

    web_ctx = runtime.context.get("mcp_context") or runtime.context.get("web_mcp_state")
    web_ctx_text = json.dumps(web_ctx, ensure_ascii=False) if web_ctx else "无"

    user_messages = filter_messages(state["messages"], include_types="human")
    last_user_msg = user_messages[-1].content if user_messages else ""

    system_prompt = (
        "你是Hub-and-Spoke中心调度路由器。需要判断购买意向等级，并基于用户问题与端侧状态进行任务路由。\n"
        "路由规则：\n"
        "- 用户要推荐/找房/看房 -> RECOMMEND\n"
        "- 用户咨询政策/税费 -> POLICY_RAG\n"
        "- 无法识别、推荐无结果、明确预约/人工 -> HANDOFF\n"
        "- 子图执行后统一进入 EMPATHY"
    )
    user_prompt = (
        f"端侧状态：{web_ctx_text}\n"
        f"用户输入：{last_user_msg}\n"
        "请输出意向等级、情绪标签和路由。"
    )
    # 部分 OpenAI 兼容网关对结构化输出支持不稳定，这里改为 JSON + 归一化兜底。
    json_prompt = (
        "请仅输出一个JSON对象，不要输出其它文本。"
        "字段必须是 intent_level, emotion_label, route_destination。"
        "intent_level 仅可为 low|medium|high；"
        "emotion_label 仅可为 positive|neutral|negative|anxious|urgent；"
        "route_destination 仅可为 RECOMMEND|POLICY_RAG|HANDOFF|EMPATHY|END。"
    )
    raw = model.invoke(
        [
            SystemMessage(content=system_prompt + "\n" + json_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    raw_obj = _extract_json_object(getattr(raw, "content", ""))

    intent_raw = raw_obj.get("intent_level") or raw_obj.get("intention_level") or raw_obj.get("intent")
    emotion_raw = raw_obj.get("emotion_label") or raw_obj.get("emotion")
    route_raw = (
        raw_obj.get("route_destination")
        or raw_obj.get("route")
        or raw_obj.get("next_agent")
        or raw_obj.get("target")
    )

    decision = EmotionRouteDecision(
        intent_level=_normalize_intent_level(intent_raw),
        emotion_label=_normalize_emotion_label(emotion_raw),
        route_destination=_normalize_route(route_raw, last_user_msg),
    )
    return {
        "intent_level": decision.intent_level,
        "emotion_label": decision.emotion_label,
        "route_destination": decision.route_destination,
        # 兼容旧字段
        "route": {
            "RECOMMEND": "recommend_subgraph",
            "POLICY_RAG": "policy_rag_subgraph",
            "HANDOFF": "handoff_subgraph",
            "EMPATHY": "empathy_generator",
            "END": "__end__",
        }.get(decision.route_destination, "handoff_subgraph"),
    }


def emotion_router_node(state: State, runtime: Runtime[ContextSchema]):
    """兼容旧调用名，内部委托给 supervisor_router_node。"""
    return supervisor_router_node(state, runtime)


def memory_manager_response(state: State):
    """当用户询问记忆时，输出总结性回复。"""
    user_messages = filter_messages(state["messages"], include_types="human")
    last_user_msg = user_messages[-1].content if user_messages else ""
    memory = state.get("user_memory", {})

    system_prompt = (
        "你是终身记忆管家，负责向用户解释其长期偏好或历史信息。"
        "如果记忆为空，说明尚未积累到足够信息。"
    )
    user_prompt = (
        f"用户问题：{last_user_msg}\n"
        f"长期记忆：{json.dumps(memory, ensure_ascii=False)}\n"
        "请给出简洁、可读的回复。"
    )
    result = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return {"messages": [result]}


def empathy_generator_node(state: State):
    """关怀与生成 Agent，输出最终话术。"""
    emotion = state.get("emotion_label", "neutral")
    intent_level = state.get("intent_level", "medium")
    memory = state.get("user_memory", {})
    profile = state.get("user_profile", {})
    outcomes = state.get("agent_outcomes", {})

    # 推荐子图严格模式：直接返回已校验结果，杜绝二次改写导致的幻觉。
    recommend_outcome = outcomes.get("recommend") if isinstance(outcomes, dict) else None
    if isinstance(recommend_outcome, dict) and recommend_outcome.get("strict_mode"):
        strict_text = str(recommend_outcome.get("final_text") or "").strip()
        if strict_text:
            return {
                "messages": [AIMessage(content=strict_text)],
                "route_destination": "END",
                "route": "__end__",
            }

    user_messages = filter_messages(state["messages"], include_types="human")
    last_user_msg = user_messages[-1].content if user_messages else ""
    last_answer = state["messages"][-1].content if state.get("messages") else ""

    system_prompt = (
        "你是关怀与生成Agent。需要基于情绪标签与意向等级，生成最终话术。"
        "- negative/anxious: 先安抚再给方案。\n"
        "- urgent: 直接、简洁、高效。\n"
        "- positive: 积极鼓励。\n"
        "保持事实准确，不要编造房源信息。"
    )
    user_prompt = (
        f"情绪标签：{emotion}\n"
        f"意向等级：{intent_level}\n"
        f"用户长期偏好：{json.dumps(memory, ensure_ascii=False)}\n"
        f"结构化画像：{json.dumps(profile, ensure_ascii=False)}\n"
        f"子图结果汇总：{json.dumps(outcomes, ensure_ascii=False)}\n"
        f"用户问题：{last_user_msg}\n"
        f"已有答复或检索结果：{last_answer}\n"
        "请生成最终回复。"
    )
    result = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return {
        "messages": [result if isinstance(result, AIMessage) else AIMessage(content=getattr(result, "content", str(result)))],
        "route_destination": "END",
        "route": "__end__",
    }

class UserMessage(BaseModel):
    type: Literal["recommend_house", "reserve_house", "get_info", "policy_rag", "others"] = Field(
        description="根据用户问题描述判断问题类型：推荐房源、预定房源、获取信息、政策咨询、其它内容"
    )

# 节点：识别用户意图
def identify_question(state: State):
    # state["messages"] # 用户问题  -》 LLM  -> 结构化输出（type） : 推荐、预定、我的、其它
    user_intent = model.with_structured_output(UserMessage).invoke(
        [SystemMessage(content="你是一个根据描述提取信息的提取专家。请从用户的描述中提取想要咨询的相关信息。"
                    "严谨根据语义推断信息，但是不能猜测或者编造信息。"
                    "类型包括：推荐房源、预定房源、获取信息、政策咨询、其它内容。"), state["messages"][-1]]
    )
    return {
        "user_intent": user_intent.type  # 条件边使用
    }

# 节点：中断询问是否需要帮助预定房源
def need_reserve(state: State, runtime: Runtime[ContextSchema]) -> NeedReserveOutput:
    if runtime.context.get("interactive_mode") in {"streamlit", "cli"}:
        decision = runtime.context.get("need_reserve_decision") or "不需要"
        if decision not in {"需要", "不需要"}:
            decision = "不需要"
        return {"reserve": decision}

    prompt = f"已经为您推荐合适的房源，是否需要帮您预订房源？\n"
    prompt += "如果不需要,请输入'**不需要**'。\n"
    prompt += "如果需要,请输入'**需要**'。\n(注意输入其它值无效)\n"
    answer = interrupt(prompt)
    return {"reserve": answer}  # 条件边获取到后，是否执行预定子图

# 节点：返回用户偏好信息
def get_user_preferences(state: State):

    # 获取最新历史偏好信息（参考答案）
    prefs = state.get("user_preferences", {})
    # 筛选用户消息（获取到用户问题）
    user_messages = filter_messages(state["messages"], include_types="human")
    reserved_info = prefs.get("reserved_info", [])
    if reserved_info:
        # 有预定过的信息
        reserved_str = "\n"
        for i, item in enumerate(reserved_info, 1):
            reserved_str += f"{i}. 预定工单ID: {item.order_id}，" \
                            f"房源标题：{item.title}，" \
                            f"预定电话：{item.phone_number}\n"
    else:
        # 没有预定
        reserved_str = "无"

    result = model.invoke(
        [SystemMessage(content="""你是一个乐于助人的助手，可以根据用户偏好信息进行回复。
如果有的偏好数据为空，不要猜测或编造数据。
不要直接回复偏好数据是什么，要结合问题进行生动回复。
如果问题与用户偏好数据无关，直接回复即可。""") ,
         HumanMessage(content="用户的历史偏好信息如下"
                      f"1. 最低预算：{prefs.get('budget_min')}"
                      f"2. 最高预算：{prefs.get('budget_max')}"
                      f"3. 已预定过的信息：{reserved_str}"
                      ) ,
         user_messages[-1]  # 问题
        ]
    )
    return {
        "messages": [result]
    }




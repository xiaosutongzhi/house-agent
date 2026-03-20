import uuid
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, ToolRuntime, InjectedStore
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.agent.common.context import ContextSchema
from src.agent.common.llm import model
from src.agent.common.store import ReservedInfo, UserPreferences
from src.agent.state.reserve import ReserveState


MISSING = "__MISSING__"


def _from_context(runtime: Runtime[ContextSchema], field: str):
    reserve_form = runtime.context.get("reserve_form") or {}
    value = reserve_form.get(field)
    if value is None:
        return None
    value = str(value).strip()
    return value or None

# 节点：获取预定房源名称
def get_title(state: ReserveState, runtime: Runtime[ContextSchema]):
    title_from_ctx = _from_context(runtime, "title")
    if title_from_ctx:
        return {"title": title_from_ctx}
    if runtime.context.get("interactive_mode") in {"streamlit", "cli"}:
        return {"title": MISSING}

    prompt = "请输入要预定的房源名称"

    while True:
        title = interrupt(prompt)
        if title:  # 验证操作
            return {"title": title}

        # 验证失败：再次输入
        prompt = f"‘{title}’ 不是一个有效的房源名称，请更正"


# 节点：获取预定电话
def get_phone(state: ReserveState, runtime: Runtime[ContextSchema]):
    phone_from_ctx = _from_context(runtime, "phone_number")
    if phone_from_ctx:
        return {"phone_number": phone_from_ctx}
    if runtime.context.get("interactive_mode") in {"streamlit", "cli"}:
        return {"phone_number": MISSING}

    prompt = "请输入要预定的手机号"

    while True:
        phone_number = interrupt(prompt)
        if phone_number:  # 验证操作
            return {"phone_number": phone_number}

        # 验证失败：再次输入
        prompt = f"‘{phone_number}’ 不是一个有效的电话号码，请更正"


# 节点：获取预定人员身份证
def get_id(state: ReserveState, runtime: Runtime[ContextSchema]):
    id_from_ctx = _from_context(runtime, "id_card")
    if id_from_ctx:
        return {"id_card": id_from_ctx}
    if runtime.context.get("interactive_mode") in {"streamlit", "cli"}:
        return {"id_card": MISSING}

    prompt = "请输入要预定的身份证号码"

    while True:
        id_card = interrupt(prompt)
        if id_card:  # 验证操作
            return {"id_card": id_card}

        # 验证失败：再次输入
        prompt = f"‘{id_card}’ 不是一个有效的身份证号码，请更正"

# 节点：更新HumanMessage
def add_reserve_message(state: ReserveState):
    missing_fields = []
    if state.get("title") in {None, "", MISSING}:
        missing_fields.append("房源标题")
    if state.get("phone_number") in {None, "", MISSING}:
        missing_fields.append("手机号")
    if state.get("id_card") in {None, "", MISSING}:
        missing_fields.append("身份证号")

    if missing_fields:
        return {
            "messages": [
                AIMessage(content=f"预定信息不完整，请补充：{', '.join(missing_fields)}。")
            ]
        }

    reserve_prompt = """
根据提供的信息，帮我预定房源。
- 预定的房源标题：{title}
- 用户预定号码：{phone_number}
- 用户身份证号码：{id_card}
    """

    return {"messages": [HumanMessage(content=reserve_prompt.format(
        title=state["title"],
        phone_number=state["phone_number"],
        id_card=state["id_card"]
    ))]}


# 1. 生成工单
# 2. 持久化存储预定信息
#    运行时数据：ToolRuntime
#    store: Annotated[Any, InjectedStore()]  在工具中注入store
@tool
def generate_orders(phone_number: str, id_card: str, house_title: str,
                    runtime: ToolRuntime, store: Annotated[Any, InjectedStore()]) -> str:
    """
根据用户电话、身份证、预定的房源，生成工单号。

Args:
    phone_number: 用户电话
    id_card：身份证
    house_title：用户要预定的房源标题
    runtime：工具的运行时信息
    store：注入工具的持久存储
    """

    # 1. 模拟生成工单号(扩展：持久化订单表)
    order_id = str(uuid.uuid4())

    # 2. 构建预定信息
    reserved_info = ReservedInfo(
        order_id=order_id,
        title=house_title,
        phone_number=phone_number
    )

    # 3. 持久化存储用户偏好（预定信息）
    if store is not None:
        user_id = runtime.context.get("user_id")
        namespace = (user_id, "preferences")
        # 查询
        prefs_result = store.search(namespace)
        if len(prefs_result) == 0:
            # 没有持久化信息，新增
            prefs = UserPreferences(
                reserved_info=[reserved_info]
            )
            store.put(
                namespace,
                str(uuid.uuid4()),
                prefs.model_dump(exclude_none=True))
        else:
            # 有偏好数据，更新
            prefs = prefs_result[0].value or {}
            prefs.setdefault('reserved_info', []).append(reserved_info)
            store.put(
                namespace,
                prefs_result[0].key,
                prefs
            )

    return f"已成功预定房源：{house_title}, 预定工单号为：{order_id}"

# 将来就可以将这个工具，使用ToolNode进行定义
tool_node = ToolNode([generate_orders])

# 节点：执行模型：1. 决定进行工具（预定房源）调用  2. 返回最终结果
def call_orders(state: ReserveState):
    if state.get("title") in {None, "", MISSING} or state.get("phone_number") in {None, "", MISSING} or state.get("id_card") in {None, "", MISSING}:
        return {
            "messages": [AIMessage(content="未执行预定：请先完整填写房源标题、手机号和身份证号。")]
        }

    return {"messages": [model.bind_tools([generate_orders]).invoke(
        [SystemMessage(content="你是一个工单生成的助手，支持调用工具进行房源预定工单生成。支持查看结果并返回最终答案")]
        + state["messages"]
    )]}


def handoff_reserve_node(state: dict, runtime: Runtime[ContextSchema]):
    """兜底/预约统一节点：挂起状态并生成 Webhook payload。"""
    user_profile = state.get("user_profile") or {}
    messages = state.get("messages") or []
    route_destination = state.get("route_destination") or "HANDOFF"
    reason = "intent_fallback"
    if route_destination == "HANDOFF":
        reason = "explicit_handoff_or_no_match"

    payload = {
        "event": "vip_house_handoff",
        "reason": reason,
        "user_id": runtime.context.get("user_id", "anonymous"),
        "user_profile": user_profile,
        "messages": [
            {
                "type": getattr(m, "type", m.__class__.__name__),
                "content": getattr(m, "content", ""),
            }
            for m in messages[-12:]
        ],
    }

    confirm_text = "已为您安排人工VIP顾问接手，并同步预约/跟进信息，请稍候。"
    return {
        "agent_outcomes": {
            **(state.get("agent_outcomes") or {}),
            "handoff": {
                "status": "queued",
                "payload": payload,
                "message": confirm_text,
            },
        },
        "messages": [AIMessage(content=confirm_text)],
    }










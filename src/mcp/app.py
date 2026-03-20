import streamlit as st
import json
import time
import redis # 👈 确保这行一定在最上面
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from src.agent.common.property_store import get_property_store

# 👈 确保这行在任何函数之外（全局变量）
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


@st.cache_resource
def load_langgraph():
    """延迟加载图，避免 Streamlit 热重载重复初始化。"""
    try:
        from agent import graph as app_graph
    except Exception:
        from src.agent import graph as app_graph
    return app_graph

@st.cache_data(ttl=60)
def load_properties() -> list[dict[str, Any]]:
    """房源大厅统一数据源：MySQL 优先，失败回退内置种子。"""
    return get_property_store().list_properties()


PROPERTIES = load_properties()


# --- 2. 初始化全局状态 (彻底抛弃老式的计数器) ---
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
    st.session_state.current_prop = PROPERTIES[0]
    st.session_state.action_trail = ["初始化进入找房大厅"] # 👈 唯一需要的记录器
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "你好，我是你的买房助手。你可以先说说预算、区域或政策问题。"}
    ]
if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_user_001"

# --- 3. 核心机制：动作记录器 ---
def log_action(action_desc):
    """每次用户交互，调用此函数把动作写进流水账"""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.action_trail.append(f"[{timestamp}] {action_desc}")
    
    # 滑动窗口：永远只保留最近的 10 个动作，防止大模型上下文爆掉
    if len(st.session_state.action_trail) > 10:
        st.session_state.action_trail.pop(0)

# --- 4. 状态同步机制 ---
def flush_state_to_redis():
    """每次页面重绘时，把最新的情况刷入 Redis"""
    state_data = {
        "business_context": {
            "viewing_property": st.session_state.current_prop["title"],
            "price": f"{st.session_state.current_prop['price']}万"
        },
        "recent_action_trail": st.session_state.action_trail 
    }
    try:
        redis_client.set("current_user_intent", json.dumps(state_data, ensure_ascii=False))
    except Exception:
        pass # 前端不报死错，避免影响展示
    return state_data


def read_state_from_redis() -> dict[str, Any]:
    """读取 Redis 中最近一次端侧状态，读不到则返回当前页面状态。"""
    try:
        value = redis_client.get("current_user_intent")
        if value:
            return json.loads(value)
    except Exception:
        pass
    return flush_state_to_redis()


def invoke_graph_with_context(
    user_text: str,
    web_state: dict[str, Any],
    *,
    need_reserve_decision: str = "不需要",
    reserve_form: dict[str, Any] | None = None,
) -> str:
    """调用 LangGraph 主图（简单版）。"""
    app_graph = load_langgraph()

    # 仅保留最近 8 轮，避免上下文过长
    recent_ui_msgs = st.session_state.chat_messages[-16:]
    lc_messages = []
    for m in recent_ui_msgs:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
    # 追加最新用户输入
    lc_messages.append(HumanMessage(content=user_text))

    payload = {"messages": lc_messages}
    context = {
        "user_id": st.session_state.user_id,
        "web_mcp_state": web_state,
        "interactive_mode": "streamlit",
        "need_reserve_decision": need_reserve_decision,
        "reserve_form": reserve_form,
    }

    try:
        result = app_graph.invoke(payload, context=context)
    except TypeError:
        # 兼容部分版本调用签名
        result = app_graph.invoke(payload)

    if isinstance(result, dict) and result.get("messages"):
        msg = result["messages"][-1]
        return getattr(msg, "content", str(msg))
    return str(result)


# --- 5. 前端 UI 构建 ---
st.set_page_config(page_title="贝壳找房-Agent版", layout="wide")
st.title("🏡 智选好房 (基于 LangGraph 隐式意向感知)")

with st.sidebar:
    st.subheader("会话设置")
    st.session_state.user_id = st.text_input("user_id", value=st.session_state.user_id)
    st.caption("同一个 user_id 会复用跨会话记忆")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏘️ 房源大厅",
    "🧮 房贷计算器",
    "🤖 智能问答",
    "📝 快速预定",
    "⚙️ 开发者监控(发给LLM的数据)",
])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("筛选房源")
        regions = list(set([p["region"] for p in PROPERTIES]))
        selected_region = st.selectbox("选择区域", ["全部"] + regions)
        filtered_props = PROPERTIES if selected_region == "全部" else [p for p in PROPERTIES if p["region"] == selected_region]
        
        for prop in filtered_props:
            # 👈 变更：点击房源时，记录轨迹
            if st.button(f"{prop['title']} ({prop['price']}万)", key=prop['id'], use_container_width=True):
                st.session_state.current_prop = prop
                log_action(f"点击查看了区域为【{prop['region']}】的房源：{prop['title']}")

    with col2:
        curr = st.session_state.current_prop
        st.subheader("当前浏览详情")
        st.info(f"**{curr['title']}**")
        st.write(f"🏷️ **总价:** {curr['price']} 万 RMB | 📐 **户型:** {curr['layout']} | **面积:** {curr['area']} 平米")
        st.image("https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", caption="室内实景展示")
        
        # 👈 变更：收藏时，记录轨迹，附加情感色彩词辅助LLM判断
        if st.button("⭐ 收藏该房源 (高意向动作)"):
            log_action(f"进行了收藏操作！对 {curr['title']} 表现出极高兴趣。")
            st.toast("收藏成功！")

with tab2:
    st.subheader("房贷计算器")
    curr = st.session_state.current_prop
    loan_amount = st.number_input("贷款总额 (万元)", value=float(curr['price'] * 0.7), step=10.0)
    years = st.selectbox("贷款年限", [10, 20, 30], index=2)
    rate = st.slider("商贷利率 (%)", 2.0, 5.0, 3.85, 0.05)
    
    # 👈 变更：计算时，记录轨迹
    if st.button("开始计算月供"):
        log_action(f"使用了房贷计算器，正在测算总价{curr['price']}万房源的还款压力，贷款{loan_amount}万。")
        monthly_rate = rate / 100 / 12
        months = years * 12
        payment = (loan_amount * 10000 * monthly_rate * ((1 + monthly_rate) ** months)) / (((1 + monthly_rate) ** months) - 1)
        st.success(f"计算完成：每月需还款 **¥{payment:,.2f}** 元")

with tab3:
    st.subheader("🤖 LangGraph 主调用界面")
    st.caption("此界面会把 Redis 中的端侧行为状态一起传给 Emotion Router Agent。")

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("请输入你的需求，例如：我在浦东预算 900 万内，想看改善三房")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        try:
            web_state = read_state_from_redis()
            answer = invoke_graph_with_context(
                prompt,
                web_state,
                need_reserve_decision="不需要",
            )
        except Exception as e:
            answer = (
                "调用图失败。可能原因：\n"
                "1) 模型/API Key 未配置；\n"
                "2) 当前流程触发了中断节点（如预定信息补全）；\n"
                f"3) 运行时错误：{e}"
            )

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("清空对话", use_container_width=True):
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "你好，我是你的买房助手。你可以先说说预算、区域或政策问题。"}
            ]
            st.rerun()
    with col_b:
        if st.button("手动刷新端侧状态", use_container_width=True):
            flush_state_to_redis()
            st.toast("已同步到 Redis")


with tab4:
    st.subheader("📝 一键预定（Streamlit 表单适配 interrupt）")
    st.caption("该表单会把预定字段直接注入图上下文，避免命令行 interrupt。")

    with st.form("reserve_form"):
        reserve_title = st.text_input("房源标题", value=st.session_state.current_prop["title"])
        reserve_phone = st.text_input("手机号", placeholder="例如 13800000000")
        reserve_id_card = st.text_input("身份证号", placeholder="例如 6101************")
        submitted = st.form_submit_button("提交预定")

    if submitted:
        web_state = read_state_from_redis()
        reserve_payload = {
            "title": reserve_title,
            "phone_number": reserve_phone,
            "id_card": reserve_id_card,
        }
        reserve_prompt = (
            f"我要预定房源：{reserve_title}。"
            f"手机号：{reserve_phone}。"
            f"身份证号：{reserve_id_card}。请直接帮我下单。"
        )
        try:
            reserve_answer = invoke_graph_with_context(
                reserve_prompt,
                web_state,
                need_reserve_decision="需要",
                reserve_form=reserve_payload,
            )
        except Exception as e:
            reserve_answer = f"预定调用失败：{e}"
        st.success(reserve_answer)


with tab5:
    # 无论上面点了什么，最后统一下发到 Redis 并展示
    current_state = flush_state_to_redis()
    st.subheader("🕵️ 实时流水账 (即将被 LangGraph 读取)")
    # st.caption("不要你写死 if/else，把这段鲜活的文字直接发给 Router Agent 综合评估！")
    st.json(current_state)
# import streamlit as st
# import json
# import time
# import pandas as pd
# import redis # 👈 确保这行一定在最上面

# # 👈 确保这行在任何函数之外（全局变量）
# redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# # --- 1. 模拟数据库：10套精选房源 ---
# PROPERTIES = [
#     # 浦东新区 (张江/陆家嘴/碧云)
#     {"id": "SH-PD-001", "title": "张江高科 汤臣豪园 满五唯一 两居", "price": 820, "layout": "2室1厅", "area": 89, "region": "浦东"},
#     {"id": "SH-PD-002", "title": "陆家嘴 世茂滨江花园 一线江景", "price": 1500, "layout": "3室2厅", "area": 145, "region": "浦东"},
#     {"id": "SH-PD-003", "title": "碧云国际社区 银杏苑 低密洋房", "price": 2100, "layout": "4室2厅", "area": 180, "region": "浦东"},
#     {"id": "SH-PD-004", "title": "三林 东方康桥 轨交沿线 刚需首选", "price": 450, "layout": "1室1厅", "area": 55, "region": "浦东"},
#     {"id": "SH-PD-005", "title": "唐镇 仁恒东郊花园 精装大平层", "price": 1280, "layout": "3室2厅", "area": 138, "region": "浦东"},
#     # 徐汇区 (徐家汇/徐汇滨江)
#     {"id": "SH-XH-006", "title": "徐家汇 汇贤居 核心地段 拎包入住", "price": 1150, "layout": "2室2厅", "area": 105, "region": "徐汇"},
#     {"id": "SH-XH-007", "title": "徐汇滨江 百汇园 高区采光极佳", "price": 1800, "layout": "4室2厅", "area": 180, "region": "徐汇"},
#     {"id": "SH-XH-008", "title": "龙华 宛平南路小区 优质双学区", "price": 980, "layout": "2室1厅", "area": 72, "region": "徐汇"},
#     # 静安区 (静安寺/大宁)
#     {"id": "SH-JA-009", "title": "静安寺 远洋豪墅 闹中取静", "price": 2200, "layout": "3室2厅", "area": 160, "region": "静安"},
#     {"id": "SH-JA-010", "title": "大宁金茂府 科技住宅 恒温恒湿", "price": 1350, "layout": "3室1厅", "area": 118, "region": "静安"},
#     {"id": "SH-JA-011", "title": "不夜城 中远两湾城 苏河景观", "price": 850, "layout": "2室2厅", "area": 98, "region": "静安"},
#     # 闵行区 (莘庄/七宝/紫竹)
#     {"id": "SH-MH-012", "title": "莘庄 世纪阳光园 轨交房", "price": 650, "layout": "2室1厅", "area": 82, "region": "闵行"},
#     {"id": "SH-MH-013", "title": "七宝 万科城市花园 优质学区", "price": 780, "layout": "2室2厅", "area": 95, "region": "闵行"},
#     {"id": "SH-MH-014", "title": "紫竹半岛 一线湖景 适宜养老", "price": 1100, "layout": "3室2厅", "area": 125, "region": "闵行"},
#     # 杨浦区 & 普陀区 & 长宁区
#     {"id": "SH-YP-015", "title": "新江湾城 仁恒怡庭 低密洋房", "price": 1420, "layout": "3室2厅", "area": 135, "region": "杨浦"},
#     {"id": "SH-YP-016", "title": "五角场 创智坊 毗邻高校 氛围好", "price": 720, "layout": "2室1厅", "area": 80, "region": "杨浦"},
#     {"id": "SH-PT-017", "title": "长风生态商务区 中海紫御豪庭", "price": 1650, "layout": "4室2厅", "area": 175, "region": "普陀"},
#     {"id": "SH-PT-018", "title": "真如 星光域 大平层 视野开阔", "price": 1200, "layout": "3室2厅", "area": 140, "region": "普陀"},
#     {"id": "SH-CN-019", "title": "中山公园 兆丰帝景苑 商圈核心", "price": 1050, "layout": "2室2厅", "area": 108, "region": "长宁"},
#     {"id": "SH-CN-020", "title": "古北 仁恒河滨花园 国际社区", "price": 1750, "layout": "3室2厅", "area": 155, "region": "长宁"}
# ]

# # --- 2. 初始化全局状态 ---
# if "start_time" not in st.session_state:
#     st.session_state.start_time = time.time()
#     st.session_state.views_count = {}
#     st.session_state.used_loan_calc = False
#     st.session_state.current_prop = PROPERTIES[0]

# # --- 3. 意向状态落盘函数 (模拟向 Redis 写入数据) ---
# def sync_intent_to_mcp():
#     time_on_page = int(time.time() - st.session_state.start_time)
#     curr_id = st.session_state.current_prop["id"]
#     views = st.session_state.views_count.get(curr_id, 1)

#     intent_level = "LOW"
#     if st.session_state.used_loan_calc:
#         intent_level = "HIGH"
#     elif views > 3 or time_on_page > 60:
#         intent_level = "MEDIUM"

#     state_data = {
#         "business_context": {
#             "current_viewing_id": curr_id,
#             "viewing_title": st.session_state.current_prop["title"],
#             "price_w": st.session_state.current_prop["price"]
#         },
#         "intent_signals": {
#             "repeated_views": views,
#             "used_loan_calculator": st.session_state.used_loan_calc,
#             "session_duration_sec": time_on_page
#         },
#         "inferred_intent_level": intent_level
#     }

#     # 🚀 架构升级：直接写入 Redis，设置一个 Key 叫 "current_user_intent"
#     try:
#         redis_client.set("current_user_intent", json.dumps(state_data, ensure_ascii=False))
#     except redis.ConnectionError:
#         st.sidebar.error("❌ Redis 连接失败，请检查服务是否启动！")
        
#     return state_data
# # --- 4. 前端 UI 构建 ---
# st.set_page_config(page_title="贝壳找房-Agent版", layout="wide")
# st.title("🏡 智选好房 (含端侧意向感知系统)")

# # 使用 Tabs 将不同功能隔离开，显得更像真实系统
# tab1, tab2, tab3 = st.tabs(["🏘️ 房源大厅", "🧮 房贷计算器", "⚙️ 开发者调试"])

# with tab1:
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         st.subheader("筛选房源")
#         # 提取区域供筛选
#         regions = list(set([p["region"] for p in PROPERTIES]))
#         selected_region = st.selectbox("选择区域", ["全部"] + regions)
        
#         filtered_props = PROPERTIES if selected_region == "全部" else [p for p in PROPERTIES if p["region"] == selected_region]
        
#         # 房源列表点击逻辑
#         for prop in filtered_props:
#             if st.button(f"{prop['title']} ({prop['price']}万)", key=prop['id'], use_container_width=True):
#                 st.session_state.current_prop = prop
#                 st.session_state.views_count[prop['id']] = st.session_state.views_count.get(prop['id'], 0) + 1

#     with col2:
#         curr = st.session_state.current_prop
#         st.subheader("当前浏览详情")
#         st.info(f"**{curr['title']}**")
#         st.write(f"🏷️ **总价:** {curr['price']} 万 RMB")
#         st.write(f"📐 **户型:** {curr['layout']} | **面积:** {curr['area']} 平米 | **区域:** {curr['region']}")
#         st.image("https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", caption="室内实景展示")
        
#         if st.button("⭐ 收藏该房源 (意向+1)"):
#             st.session_state.views_count[curr['id']] = st.session_state.views_count.get(curr['id'], 0) + 2
#             st.toast("收藏成功！")

# with tab2:
#     st.subheader("房贷计算器 (核心意向埋点)")
#     st.caption("在真实业务中，用户一旦使用计算器，说明进入了实质性的比价筹款阶段。")
#     loan_amount = st.number_input("贷款总额 (万元)", value=st.session_state.current_prop['price'] * 0.7, step=10.0)
#     years = st.selectbox("贷款年限", [10, 20, 30], index=2)
#     rate = st.slider("商贷利率 (%)", 2.0, 5.0, 3.85, 0.05)
    
#     if st.button("开始计算"):
#         st.session_state.used_loan_calc = True # 核心状态改变！
#         monthly_rate = rate / 100 / 12
#         months = years * 12
#         payment = (loan_amount * 10000 * monthly_rate * ((1 + monthly_rate) ** months)) / (((1 + monthly_rate) ** months) - 1)
#         st.success(f"计算完成：每月需还款 **¥{payment:,.2f}** 元")

# with tab3:
#     current_state = sync_intent_to_mcp()
#     st.subheader("实时状态监控面板 (MCP Payload)")
#     st.json(current_state)
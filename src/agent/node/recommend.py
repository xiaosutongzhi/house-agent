import os
import random
import uuid
import json
import ast
import re
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.agent.common.context import ContextSchema
from src.agent.common.llm import model
from src.agent.common.property_store import get_property_store
from src.agent.common.store import UserPreferences
from src.agent.state.recommend import RecommendState, get_recommend_info


def _clean_env_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip('"').strip("'")


def _is_db_disabled() -> bool:
    if os.getenv("DISABLE_DB", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    required = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
    return any(not _clean_env_value(os.getenv(k)) for k in required)


class UserInfo(BaseModel):
    """用户的租房需求信息"""

    city: Optional[str] = Field(
        default=None,
        description="用户所在或想要租房的城市，例如：西安、北京、上海"
    )
    district: Optional[str] = Field(
        default=None,
        description="用户想要租房的具体区域或行政区，例如：雁塔区、碑林区、海淀区"
    )
    budget_min: Optional[float] = Field(
        default=None,
        description="用户的最低预算，单位为元/月。如果是xx元以内，要设置最小值为0"
    )
    budget_max: Optional[float] = Field(
        default=None,
        description="用户的最高预算，单位为元/月。如果是xx元以上，最大值设置为10000"
    )
    room_type: Optional[str] = Field(
        default=None,
        description="房屋类型，例如：整租、合租、公寓、一室一厅、两室一厅"
    )
    orientation: Optional[str] = Field(
        default=None,
        description="房屋朝向，例如：朝南、朝北、东南、南北通透"
    )
    room_count: Optional[int] = Field(
        default=None,
        description="需要推荐的房屋数量"
    )
    others: Optional[str] = Field(
        default=None,
        description="特殊要求，例如：带阳台、独立卫生间、近地铁、可养宠物、有电梯等"
    )


def collect_user_info(state: RecommendState, runtime: Runtime[ContextSchema], *, store: BaseStore | None = None):
    """收集用户希望的推荐信息"""

    # 1. 获取需要被解析的数据：最新的用户消息 + 用户的偏好数据
    # 场景1：
    # 最新的用户消息： 西安 3套
    # 用户的偏好数据： 预算 1000-2000元
    user_messages = filter_messages(state["messages"], include_types="human")
    pref = state.get("user_preferences")
    if pref and (pref["budget_min"] or pref["budget_max"]):
        # 偏好中包含最低和最高预算
        extract_messages = [
            HumanMessage(content="用户的历史偏好信息如下："
                         f"1. 最低预算：{pref['budget_min']}"
                         f"2. 最高预算：{pref['budget_max']}"),
            user_messages[-1]
        ]
    else:
        # 无偏好数据
        extract_messages = [user_messages[-1]]

    # 2. 提取信息(LLM结构化返回)
    # 拓展：将信息与数据库表中的字段进行映射
    def extract_info(messages) -> UserInfo:
        system_message = SystemMessage(
            content="""
你是一个租房需求信息提取专家。请从用户的描述与历史信息中提取租房相关信息。
如果用户历史偏好信息与最新用户消息冲突，以最新的用户消息为主。
只提取用户明确提到的信息，不要猜测或推断。
如果某个信息用户没有提到，就返回null。
注意预算的单位可能是元/月、元/天等，请统一转换为元/月。
如果用户提到价格范围，请分别提取最低和最高预算。
如果用户提到推荐几套房，提取room_count字段。"""
        )
        return model.with_structured_output(schema=UserInfo).invoke([system_message] + messages)

    # 更新状态函数
    def update_state(current_state: dict, info: UserInfo) -> dict:
        if not info:
            return current_state

        user_info_dict = info.model_dump(exclude_none=True)
        current_state.update(user_info_dict)
        return current_state


    # 根据历史偏好和用户消息提取消息
    updated_state = {}
    extracted_info = extract_info(extract_messages)
    updated_state = update_state(updated_state, extracted_info)

    # 3. 中断咨询推荐的必须参数
    # 场景2：
    # 最新的用户消息：给我推荐房子（并未表明推荐城市，模糊推荐）
    # 询问用户意向城市

    # 检查是否缺失关键信息: 城市、预算范围
    missing_info = []
    if not updated_state.get("city"):
        missing_info.append("**城市**")
    if updated_state.get("budget_min") is None or updated_state.get("budget_max") is None:
        missing_info.append("**预算范围**")

    if missing_info:
        prompt = f"为了给您推荐合适的房源，请提供以下信息:{'，'.join(missing_info)}和其它信息。\n"
        prompt += "如果您不想提供，请输入'**不提供**',我会根据已有信息为您推荐房源。"
        # 根据缺失的信息进行中断
        answer = interrupt(prompt)
        if str(answer).strip() == "不提供":
            # 已经缺失关键信息，而且用户还不提供。需要给关键信息设置默认值
            if not updated_state.get("city"):
                updated_state["city"] = "随机城市"
            if not updated_state.get("budget_min"):
                updated_state["budget_min"] = 500.0
            if not updated_state.get("budget_max"):
                updated_state["budget_max"] = 5000.0
            if not updated_state.get("room_count"):
                updated_state["room_count"] = 5
        else:
            # 缺失关键信息，但用户已经补充
            # 将answer构建为HumanMessage
            user_response_msg = HumanMessage(content=str(answer))
            extracted_info = extract_info([user_response_msg])
            # updated_state就是包含了中断的结果
            updated_state = update_state(updated_state, extracted_info)

    # 4. 持久化处理：更新预算
    # 场景3：
    # 最新的用户消息：西安 3套  预算0-5000元
    # 用户的偏好数据：预算 1000-2000元
    if store is not None and (updated_state.get("budget_min") or updated_state.get("budget_max")):
        # 有可能会更新
        user_id = runtime.context.get("user_id")
        namespace = (user_id, "preferences")
        prefs_result = store.search(namespace)
        if len(prefs_result) == 0:
            # 新增
            prefs = UserPreferences(
                budget_min=updated_state.get("budget_min"),
                budget_max=updated_state.get("budget_max"),
            )
            store.put(namespace,
                      str(uuid.uuid4()),
                      prefs.model_dump(exclude_none=True))
            updated_state["user_preferences"] = prefs.model_dump(exclude_none=True)
        else:
            # 有持久化信息，判断更新
            # store:  1000-5000
            # state:  2000-3000   不用更新
            # state:  500-6000    需要更新  store:  500-6000
            prefs = prefs_result[0].value
            store_min = prefs["budget_min"]
            store_max = prefs["budget_max"]
            cur_min = updated_state.get("budget_min")
            cur_max = updated_state.get("budget_max")
            update_min = False   # 是否更新最小预算
            update_max = False   # 是否更新最大预算
            if store_min is not None and cur_min is not None and cur_min < store_min:
                # 都不为空，就比较
                update_min = True
            elif store_min is None and cur_min is not None:
                # store 没有，但 cur 有
                update_min = True

            if store_max is not None and cur_max is not None and cur_max > store_max:
                update_max = True
            elif store_max is None and cur_max is not None:
                update_max = True

            if update_min or update_max:
                if update_min:
                    prefs["budget_min"] = cur_min
                if update_max:
                    prefs["budget_max"] = cur_max
                # 更新操作
                store.put(
                    namespace,
                    prefs_result[0].key, # 根据查询到的key进行更新
                    prefs
                )
                updated_state["user_preferences"] = prefs

    # 5. 准备最终的消息，并更新
    updated_state["messages"] = [HumanMessage(content=get_recommend_info(updated_state))]

    print(f"已收集用户信息：\n城市：{updated_state.get('city')}"
          f"区域：{updated_state.get('district')}"
          f"预算：{updated_state.get('budget_min')}-{updated_state.get('budget_max')}元/月"
          f"房间数：{updated_state.get('room_count')}")

    # 返回：覆盖了推荐参数和偏好数据，更新了消息列表
    return updated_state


# 使用.env环境变量(win)
load_dotenv()
_DB_DISABLED = _is_db_disabled()

if not _DB_DISABLED:
    db_user = _clean_env_value(os.getenv('DB_USER'))
    db_password = _clean_env_value(os.getenv('DB_PASSWORD'))
    db_host = _clean_env_value(os.getenv('DB_HOST'))
    db_port = _clean_env_value(os.getenv('DB_PORT'))
    db_name = _clean_env_value(os.getenv('DB_NAME'))
    try:
        db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
        # 获取数据库工具
        toolkit = SQLDatabaseToolkit(db=db, llm=model)
        tools = toolkit.get_tools()
    except Exception:
        _DB_DISABLED = True
        db = None
        tools = []
else:
    db = None
    tools = []
# print(tools)
# for tool in tools:
#     print(tool.name)
# 与数据库交互的工具
# [
#     sql_db_query: QuerySQLDatabaseTool(description="Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.", db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x00000199927617F0>),
#     sql_db_schema: InfoSQLDatabaseTool(description='Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3', db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x00000199927617F0>),
#     sql_db_list_tables: ListSQLDatabaseTool(db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x00000199927617F0>),
#     sql_db_query_checker: QuerySQLCheckerTool(description='Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!', db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x00000199927617F0>, llm=ChatOpenAI(client=<openai.resources.chat.completions.completions.Completions object at 0x0000019992093230>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x0000019992093CB0>, root_client=<openai.OpenAI object at 0x0000019992090980>, root_async_client=<openai.AsyncOpenAI object at 0x0000019992093A10>, model_name='gpt-4o-mini', temperature=0.0, model_kwargs={}, openai_api_key=SecretStr('**********'), openai_api_base='https://api.chatanywhere.tech'), llm_chain=LLMChain(verbose=False, prompt=PromptTemplate(input_variables=['dialect', 'query'], input_types={}, partial_variables={}, template='\n{query}\nDouble check the {dialect} query above for common mistakes, including:\n- Using NOT IN with NULL values\n- Using UNION when UNION ALL should have been used\n- Using BETWEEN for exclusive ranges\n- Data type mismatch in predicates\n- Properly quoting identifiers\n- Using the correct number of arguments for functions\n- Casting to the correct data type\n- Using the proper columns for joins\n\nIf there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.\n\nOutput the final SQL query only.\n\nSQL Query: '), llm=ChatOpenAI(client=<openai.resources.chat.completions.completions.Completions object at 0x0000019992093230>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x0000019992093CB0>, root_client=<openai.OpenAI object at 0x0000019992090980>, root_async_client=<openai.AsyncOpenAI object at 0x0000019992093A10>, model_name='gpt-4o-mini', temperature=0.0, model_kwargs={}, openai_api_key=SecretStr('**********'), openai_api_base='https://api.chatanywhere.tech'), output_parser=StrOutputParser(), llm_kwargs={}))
# ]

if not _DB_DISABLED:
    # 节点：获取表信息
    get_schema_tool =  next(tool for tool in tools if tool.name == "sql_db_schema")
    get_schema_node = ToolNode([get_schema_tool], name="get_schema")  # 工具执行节点（返回ToolMessage）
    # 节点：执行sql查询
    run_query_tool =  next(tool for tool in tools if tool.name == "sql_db_query")
    run_query_node = ToolNode([run_query_tool], name="run_query")     # 工具执行节点（返回ToolMessage）
else:
    get_schema_tool = None
    run_query_tool = None
    get_schema_node = ToolNode([], name="get_schema")
    run_query_node = ToolNode([], name="run_query")

# 节点：获取全量表
def list_tables(state: RecommendState):
    if _DB_DISABLED:
        return {"messages": [AIMessage(content="数据库已禁用，无法列出表。") ]}
    # 1. 获取AIMessage(tool_calls)
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "123123",
        "type": "tool_call",
    }
    # 模拟必定调用工具
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    # 2. 手动调用工具：sql_db_list_tables
    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)

    # 3. 整合结果
    response = AIMessage(content=f"可用的表：{tool_message.content}")
    return {
        "messages": [tool_call_message, tool_message, response]
    }

# 节点：绑定工具（获取表信息），让LLM将来必定执行工具节点
def call_get_schema(state: RecommendState):
    if _DB_DISABLED:
        return {"messages": [AIMessage(content="数据库已禁用，跳过获取表结构。") ]}
    llm_with_tools =  model.bind_tools([get_schema_tool], tool_choice="any")
    # llm 根据历史消息筛选表，并获取需要的表信息
    response =  llm_with_tools.invoke(state["messages"])  # AIMessage(tool_calls)
    # 下一步一定是会调用get_schema_tool工具的！
    return {
        "messages": [response]
    }


# 构造SQL: select * from house where city=西安 and 条件2=？

# 节点：生成SQL + 整合结果
def generate_query(state: RecommendState):
    if _DB_DISABLED:
        return {"messages": [AIMessage(content="数据库已禁用，无法生成查询。") ]}
    generate_query_system_prompt = """
您是一个设计用于与SQL数据库交互的代理。
给定一个输入问题，创建一个语法正确的{dialect}查询来运行，然后查看查询的结果并返回答案。
需要根据rows from table的示例设置真实查询的值。
除非用户指定了他们希望获得的特定数量的示例，否则始终将查询限制为最多{top_k}个结果。
您可以按相关列对结果排序，以返回最感兴趣的结果。不要查询特定表中的所有列，只查询给定问题的相关列。
不要对数据库做任何DML语句（INSERT， UPDATE， DELETE， DROP等)。
        """
    system_prompt = generate_query_system_prompt.format(
        dialect=db.dialect,
        top_k=state.get("room_count", 5)
    )

    system_message = SystemMessage(content=system_prompt)
    llm_with_tools = model.bind_tools([run_query_tool])
    return {
        "messages": [llm_with_tools.invoke([system_message] + state["messages"])]  # AIMessage(tool_call?)
    }

    # QuerySQLDatabaseTool
    # tool_call: {
    #     "name": sql_db_query  # 运行sql
    #     "args": {"query": "select ....."}              # sql
    #     "id":
    # }

def check_query(state: RecommendState):
    if _DB_DISABLED:
        return {"messages": [AIMessage(content="数据库已禁用，跳过SQL校验。") ]}
    # 走到这个节点，上一个节点必然是generate_query，并且最新的消息必然是AIMessage(tool_call)
    check_query_system_prompt = """
你是一个非常注重细节的SQL专家。仔细检查{dialect}查询中的常见错误，包括：
-使用NULL值的NOT IN
-在应该使用UNION ALL时使用UNION
-使用BETWEEN表示独占范围
-谓词中的数据类型不匹配
-正确引用标识符
-使用正确数量的函数参数
-转换为正确的数据类型
-使用合适的列进行连接
如果存在上述任何错误，请重写查询。如果没有错误，只需复制原始查询即可。
在运行此检查之后，您将调用适当的工具来执行查询。
        """.format(dialect=db.dialect)
    system_message = SystemMessage(content=check_query_system_prompt)
    # 将SQL当作用户消息传入进行检查
    tool_call = state["messages"][-1].tool_calls[0]
    # tool_call["args"]["query"] 就能拿到SQL
    # tool_call: {
    #     "name": sql_db_query  # 运行sql
    #     "args": {"query": "select ....."}              # sql
    #     "id":
    # }
    user_message = HumanMessage(content=tool_call["args"]["query"])
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")  # 必须让AIMessage带上tool_calls
    response = llm_with_tools.invoke([system_message, user_message])  # 带tool_calls的AIMessage
    # 目前最新的一个消息是AI(t)，现在又生成了一个AI(t)，则可以将两个合成一个
    response.id = state["messages"][-1].id
    return {"messages": [response]}


# -----------------------
# Hub-and-Spoke 推荐子图（官方 ReAct）
# -----------------------

@tool
def SQL_Search_Tool(
    budget_min: float | None = None,
    budget_max: float | None = None,
    city: str | None = None,
    bedrooms: int | None = None,
    district: str | None = None,
) -> dict[str, Any]:
    """按硬约束检索房源，返回 candidate_ids 与候选明细。"""
    store = get_property_store()
    return store.search_sql(
        budget_min=budget_min,
        budget_max=budget_max,
        city=city,
        bedrooms=bedrooms,
        district=district,
    )


@tool
def Vector_Search_Tool(
    query_soft_prefs: str,
    candidate_ids: list[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """在 candidate_ids 范围内做软偏好语义精排。"""
    store = get_property_store()
    return store.vector_search(
        query_soft_prefs=query_soft_prefs,
        candidate_ids=candidate_ids,
        top_k=top_k,
    )


def recommend_llm_node(state: dict, runtime: Runtime[ContextSchema]):
    """ReAct 的 LLM 决策节点：决定是否继续调用工具。"""
    user_profile = dict(state.get("user_profile", {}) or {})
    mcp_context = runtime.context.get("mcp_context") or runtime.context.get("web_mcp_state") or {}
    user_messages = filter_messages(state.get("messages") or [], include_types="human")
    last_user_text = str(user_messages[-1].content) if user_messages else ""
    store = get_property_store()
    all_rows = store.list_properties()

    def _find_row_by_title(title: str) -> dict[str, Any] | None:
        q = str(title or "").strip()
        if not q:
            return None
        for row in all_rows:
            row_title = str(row.get("title") or "")
            if row_title and (q in row_title or row_title in q):
                return row
        return None

    def _extract_user_interest_rows(text: str) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in all_rows:
            title = str(row.get("title") or "")
            rid = str(row.get("id") or "")
            if not title or not rid:
                continue
            if title in text and rid not in seen:
                seen.add(rid)
                found.append(row)
        return found

    soft_preferences = list(user_profile.get("soft_preferences") or [])
    focus_property = None
    mcp_interest_row: dict[str, Any] | None = None
    if isinstance(mcp_context, dict):
        focus_property = mcp_context.get("focus_property")
        if not focus_property:
            biz = mcp_context.get("business_context")
            if isinstance(biz, dict):
                focus_property = biz.get("viewing_property")
                raw_price = str(biz.get("price") or "")
                m = re.search(r"(\d+(?:\.\d+)?)", raw_price)
                if m:
                    try:
                        user_profile["mcp_interested_property_price"] = float(m.group(1))
                    except Exception:
                        pass
        if focus_property:
            soft_preferences.append(f"关注楼盘:{focus_property}")

    if focus_property:
        user_profile["mcp_interested_property_title"] = str(focus_property)
        mcp_interest_row = _find_row_by_title(str(focus_property))
        if mcp_interest_row:
            user_profile["mcp_interested_property_id"] = str(mcp_interest_row.get("id") or "")
            user_profile["mcp_interested_property_title"] = str(mcp_interest_row.get("title") or focus_property)
            user_profile["mcp_interested_property_price"] = float(mcp_interest_row.get("price") or 0)
            user_profile["mcp_interested_property_features"] = list(mcp_interest_row.get("features") or [])

    user_interest_rows = _extract_user_interest_rows(last_user_text)
    if user_interest_rows:
        user_profile["user_input_interested_property_ids"] = [str(r.get("id")) for r in user_interest_rows if r.get("id")]
        user_profile["user_input_interested_property_titles"] = [
            str(r.get("title")) for r in user_interest_rows if r.get("title")
        ]
        for r in user_interest_rows:
            t = str(r.get("title") or "")
            if t:
                soft_preferences.append(f"用户提及楼盘:{t}")

    # 兼容已有字段：优先 MCP，再用用户主动提及
    primary_interest = mcp_interest_row or (user_interest_rows[0] if user_interest_rows else None)
    if primary_interest:
        user_profile["interested_property_id"] = str(primary_interest.get("id") or "")
        user_profile["interested_property_title"] = str(primary_interest.get("title") or "")
        user_profile["interested_property_price"] = float(primary_interest.get("price") or 0)
        user_profile["interested_property_features"] = list(primary_interest.get("features") or [])

    user_profile["soft_preferences"] = list(dict.fromkeys([x for x in soft_preferences if x]))

    system_prompt = (
        "你是VIP购房导购推荐专家，使用 ReAct 决策。\n"
        "你必须严格遵循工具调用顺序：\n"
        "1) 必须先调用 SQL_Search_Tool 获取 candidate_ids（硬约束过滤）；\n"
        "2) 再调用 Vector_Search_Tool，并且必须把上一步 candidate_ids 原样传入进行语义匹配；\n"
        "3) 若工具结果不足，允许再次调用 SQL_Search_Tool 放宽约束后重试；\n"
        "4) 当你不再调用工具时，直接输出最终推荐话术（简洁列出推荐理由）。"
    )
    context_prompt = (
        f"当前结构化画像: {user_profile}\n"
        f"隐式交互上下文: {mcp_context}\n"
        f"MCP兴趣房源: {user_profile.get('mcp_interested_property_title') or '无'}\n"
        f"用户主动提及房源: {user_profile.get('user_input_interested_property_titles') or []}\n"
        f"软偏好建议词: {';'.join(soft_preferences)}"
    )

    llm_with_tools = model.bind_tools([SQL_Search_Tool, Vector_Search_Tool])
    response = llm_with_tools.invoke(
        [
            SystemMessage(content=system_prompt),
            SystemMessage(content=context_prompt),
        ]
        + state.get("messages", [])
    )
    return {"messages": [response], "user_profile": user_profile}


def recommend_finalize_node(state: dict):
    """ReAct 结束节点：只写 recommend outcome，不在子图内决定下一跳。"""
    messages = state.get("messages") or []
    final_ai: AIMessage | None = None
    tool_observations: list[dict[str, Any]] = []

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            final_ai = msg
            break

    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_observations.append(
                {
                    "tool_name": getattr(msg, "name", None),
                    "content": msg.content,
                }
            )

    def _parse_tool_content(content: Any) -> Any:
        if isinstance(content, (dict, list)):
            return content
        if not isinstance(content, str):
            return None
        text = content.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

    def _safe_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    latest_sql: dict[str, Any] = {}
    latest_vec: list[dict[str, Any]] = []
    for obs in tool_observations:
        name = obs.get("tool_name")
        parsed = _parse_tool_content(obs.get("content"))
        if name == "SQL_Search_Tool" and isinstance(parsed, dict):
            latest_sql = parsed
        elif name == "Vector_Search_Tool" and isinstance(parsed, list):
            latest_vec = [x for x in parsed if isinstance(x, dict)]

    store = get_property_store()
    db_rows = store.list_properties()
    by_id = {str(x.get("id")): x for x in db_rows if x.get("id")}
    title_rows = [x for x in db_rows if x.get("title")]

    user_profile = state.get("user_profile") or {}
    user_messages = filter_messages(messages, include_types="human")
    last_user_text = str(user_messages[-1].content) if user_messages else ""

    def _row_text(row: dict[str, Any]) -> str:
        return f"{row.get('title', '')} {' '.join(row.get('features') or [])} {row.get('district', '')}"

    def _find_row_by_id_or_title(pid: str | None = None, title: str | None = None) -> dict[str, Any] | None:
        if pid:
            row = by_id.get(str(pid))
            if row:
                return row
        q = str(title or "").strip()
        if q:
            for r in title_rows:
                t = str(r.get("title") or "")
                if t and (q in t or t in q):
                    return r
        return None

    mcp_interest_row = _find_row_by_id_or_title(
        user_profile.get("mcp_interested_property_id"),
        user_profile.get("mcp_interested_property_title"),
    )

    user_interest_rows: list[dict[str, Any]] = []
    user_interest_seen: set[str] = set()
    for pid in user_profile.get("user_input_interested_property_ids") or []:
        row = _find_row_by_id_or_title(pid, None)
        rid = str(row.get("id") or "") if row else ""
        if row and rid and rid not in user_interest_seen:
            user_interest_seen.add(rid)
            user_interest_rows.append(row)
    for title in user_profile.get("user_input_interested_property_titles") or []:
        row = _find_row_by_id_or_title(None, title)
        rid = str(row.get("id") or "") if row else ""
        if row and rid and rid not in user_interest_seen:
            user_interest_seen.add(rid)
            user_interest_rows.append(row)

    # 兼容老字段（兜底）
    if not mcp_interest_row and not user_interest_rows:
        legacy_row = _find_row_by_id_or_title(
            user_profile.get("interested_property_id"),
            user_profile.get("interested_property_title"),
        )
        if legacy_row:
            user_interest_rows.append(legacy_row)

    interest_row = mcp_interest_row or (user_interest_rows[0] if user_interest_rows else None)
    all_interest_rows = []
    all_seen: set[str] = set()
    for row in ([mcp_interest_row] if mcp_interest_row else []) + user_interest_rows:
        rid = str(row.get("id") or "")
        if rid and rid not in all_seen:
            all_seen.add(rid)
            all_interest_rows.append(row)

    def _infer_hard_constraints() -> dict[str, Any]:
        infer_bedrooms = user_profile.get("bedrooms")
        if re.search(r"(两\s*房|2\s*房|二\s*房|两\s*室|2\s*室|二\s*室)", last_user_text):
            infer_bedrooms = 2
        elif re.search(r"(三\s*房|3\s*房|三\s*室|3\s*室)", last_user_text):
            infer_bedrooms = 3

        infer_city = user_profile.get("city")
        if not infer_city and "广州" in last_user_text:
            infer_city = "广州"

        infer_budget_min = user_profile.get("budget_min")
        infer_budget_max = user_profile.get("budget_max")
        m1 = re.search(r"(\d+(?:\.\d+)?)\s*万\s*(以下|以内)", last_user_text)
        if m1:
            infer_budget_min = 0
            infer_budget_max = float(m1.group(1))
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*[-到~]\s*(\d+(?:\.\d+)?)\s*万", last_user_text)
        if m2:
            infer_budget_min = float(m2.group(1))
            infer_budget_max = float(m2.group(2))

        district = user_profile.get("district")
        if not district and interest_row:
            district = interest_row.get("district")

        return {
            "budget_min": infer_budget_min,
            "budget_max": infer_budget_max,
            "city": infer_city,
            "bedrooms": infer_bedrooms,
            "district": district,
        }

    hard_constraints = _infer_hard_constraints()

    candidate_ids = [str(x) for x in (latest_sql.get("candidate_ids") or []) if str(x)]
    candidate_set = set(candidate_ids)

    # 严格校验：仅接受来自工具结果且与 DB 中 id+price 一致的条目
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in latest_vec:
        pid = str(item.get("id") or "")
        if not pid or pid in seen_ids:
            continue
        if candidate_set and pid not in candidate_set:
            continue
        db_item = by_id.get(pid)
        if not db_item:
            continue
        db_price = _safe_float(db_item.get("price"))
        item_price = _safe_float(item.get("price"))
        # 如果工具返回了 price，必须与 DB 价格一致（允许极小浮点误差）
        if item_price is not None and db_price is not None and abs(item_price - db_price) > 1e-6:
            continue
        seen_ids.add(pid)
        validated.append(db_item)
        if len(validated) >= 3:
            break

    # 若向量结果不可用，但 SQL 有候选，回退到 SQL 候选（仍来自工具链）
    if not validated and candidate_ids:
        for pid in candidate_ids:
            if pid in seen_ids:
                continue
            db_item = by_id.get(pid)
            if not db_item:
                continue
            seen_ids.add(pid)
            validated.append(db_item)
            if len(validated) >= 3:
                break

    has_exact_match = bool(candidate_ids)

    # 若工具链没有给出 candidate_ids，尝试基于历史画像 + 本轮用户问句做确定性兜底检索
    if not validated and not has_exact_match:
        infer_soft_prefs = list(user_profile.get("soft_preferences") or [])
        if "宠物" in last_user_text and "可养宠" not in infer_soft_prefs:
            infer_soft_prefs.append("可养宠")
        if any(x in last_user_text for x in ["阳光", "采光", "南向"]):
            infer_soft_prefs.append("采光")

        inferred_sql = store.search_sql(
            budget_min=hard_constraints.get("budget_min"),
            budget_max=hard_constraints.get("budget_max"),
            city=hard_constraints.get("city"),
            bedrooms=hard_constraints.get("bedrooms"),
            district=hard_constraints.get("district"),
        )
        inferred_ids = [str(x) for x in (inferred_sql.get("candidate_ids") or []) if str(x)]
        if inferred_ids:
            inferred_vec = store.vector_search(
                query_soft_prefs=";".join(infer_soft_prefs) if infer_soft_prefs else "",
                candidate_ids=inferred_ids,
                top_k=3,
            )

            for item in inferred_vec:
                pid = str(item.get("id") or "")
                if not pid or pid in seen_ids:
                    continue
                db_item = by_id.get(pid)
                if not db_item:
                    continue
                db_price = _safe_float(db_item.get("price"))
                item_price = _safe_float(item.get("price"))
                if item_price is not None and db_price is not None and abs(item_price - db_price) > 1e-6:
                    continue
                seen_ids.add(pid)
                validated.append(db_item)
                if len(validated) >= 3:
                    break

            if not validated:
                for pid in inferred_ids:
                    if pid in seen_ids:
                        continue
                    db_item = by_id.get(pid)
                    if not db_item:
                        continue
                    seen_ids.add(pid)
                    validated.append(db_item)
                    if len(validated) >= 3:
                        break

            has_exact_match = bool(validated)

    # ------- 多情景拆分推荐（每个情景单独给 2 套） -------
    scenario_rules: list[tuple[str, str, list[str]]] = [
        ("sunlight", "阳光采光", ["阳光", "采光", "南向", "朝南", "南北通透"]),
        ("balcony", "阳台", ["阳台", "双阳台", "大阳台", "生活阳台"]),
        ("pet", "养宠", ["宠物", "养宠", "可养宠", "可养猫", "可养狗", "猫", "狗"]),
    ]

    soft_pref_text = " ".join(str(x) for x in (user_profile.get("soft_preferences") or []))
    source_text = f"{last_user_text} {soft_pref_text}"
    for ir in all_interest_rows:
        source_text += " " + _row_text(ir)

    active_scenarios: list[tuple[str, str, list[str]]] = []
    for key, label, kws in scenario_rules:
        if any(k in source_text for k in kws):
            active_scenarios.append((key, label, kws))

    base_candidate_ids = candidate_ids[:]
    if not base_candidate_ids:
        inferred = store.search_sql(
            budget_min=hard_constraints.get("budget_min"),
            budget_max=hard_constraints.get("budget_max"),
            city=hard_constraints.get("city"),
            bedrooms=hard_constraints.get("bedrooms"),
            district=hard_constraints.get("district"),
        )
        base_candidate_ids = [str(x) for x in (inferred.get("candidate_ids") or []) if str(x)]

    base_rows = [by_id[i] for i in base_candidate_ids if i in by_id]

    def _match_keywords(row: dict[str, Any], kws: list[str]) -> bool:
        text = _row_text(row)
        return any(k in text for k in kws)

    def _scenario_pick(kws: list[str], *, top_n: int = 2) -> list[dict[str, Any]]:
        if not base_candidate_ids:
            return []

        query = ";".join(kws)
        vec = store.vector_search(query_soft_prefs=query, candidate_ids=base_candidate_ids, top_k=max(6, top_n * 3))
        chosen: list[dict[str, Any]] = []
        chosen_ids: set[str] = set()

        # 先收取关键词真匹配
        for it in vec:
            pid = str(it.get("id") or "")
            if not pid or pid in chosen_ids or pid not in by_id:
                continue
            row = by_id[pid]
            if not _match_keywords(row, kws):
                continue
            chosen_ids.add(pid)
            chosen.append(row)
            if len(chosen) >= top_n:
                return chosen

        # 再补齐：候选集中关键词匹配
        for row in base_rows:
            pid = str(row.get("id") or "")
            if not pid or pid in chosen_ids:
                continue
            if not _match_keywords(row, kws):
                continue
            chosen_ids.add(pid)
            chosen.append(row)
            if len(chosen) >= top_n:
                return chosen

        return chosen

    scenario_results: list[dict[str, Any]] = []
    if active_scenarios:
        for _, label, kws in active_scenarios:
            picks = _scenario_pick(kws, top_n=2)
            scenario_results.append({"scenario": label, "items": picks, "keywords": kws})

    # 同时满足全部情景的候选（用于给出“无法同时满足”的解释）
    combined_items: list[dict[str, Any]] = []
    if len(active_scenarios) >= 2 and base_rows:
        for row in base_rows:
            if all(_match_keywords(row, kws) for _, _, kws in active_scenarios):
                combined_items.append(row)
        combined_items = combined_items[:2]

    def _similar_to_interest(top_n: int = 3) -> list[dict[str, Any]]:
        if not all_interest_rows:
            return []
        interest_ids = {str(x.get("id") or "") for x in all_interest_rows}

        pool = base_rows or db_rows
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in pool:
            rid = str(row.get("id") or "")
            if not rid or rid in interest_ids:
                continue
            score = 0.0
            r_text = _row_text(row)
            for ir in all_interest_rows:
                i_price = _safe_float(ir.get("price"))
                i_bed = ir.get("bedrooms")
                i_dist = ir.get("district") or ir.get("region")
                if i_dist and (row.get("district") == i_dist or row.get("region") == i_dist):
                    score += 2.0
                if i_bed and row.get("bedrooms") == i_bed:
                    score += 1.5
                rp = _safe_float(row.get("price"))
                if i_price is not None and rp is not None:
                    score += max(0.0, 1.5 - abs(i_price - rp) / 120.0)
                i_text = _row_text(ir)
                same_kw = 0
                for kw in ["阳台", "采光", "南向", "地铁", "可养宠", "可养猫", "南北通透"]:
                    if kw in i_text and kw in r_text:
                        same_kw += 1
                score += float(same_kw)
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_n]]

    similar_interest_items = _similar_to_interest(3)

    fallback_pool = [x for x in db_rows if str(x.get("id")) not in seen_ids]
    fallback_items: list[dict[str, Any]] = []
    if not has_exact_match and fallback_pool:
        k = min(3, len(fallback_pool))
        fallback_items = random.sample(fallback_pool, k=k)

    def _fmt_item(x: dict[str, Any]) -> str:
        return (
            f"- {x.get('title', '')}（ID:{x.get('id')}）"
            f"｜总价:{x.get('price')}万｜{x.get('layout')}｜{x.get('area')}㎡"
            f"｜区域:{x.get('district') or x.get('region') or ''}"
        )

    if scenario_results:
        lines: list[str] = []
        if len(active_scenarios) >= 2 and not combined_items:
            lines.append("没有找到可同时满足您多个情景需求的单一房源，我按每个情景分别给您推荐，方便对比决策：")
        elif combined_items:
            lines.append("找到了可同时覆盖您多个情景诉求的房源：")
            lines.extend(_fmt_item(x) for x in combined_items)

        if mcp_interest_row:
            lines.append(
                f"我看到您在页面浏览了「{mcp_interest_row.get('title')}」，下面也结合该房源给出类似推荐。"
            )
        if user_interest_rows:
            user_titles = "、".join(str(x.get("title")) for x in user_interest_rows[:3])
            lines.append(f"您在对话中主动提到的房源有：{user_titles}。我已一并纳入推荐。")

        for g in scenario_results:
            group_items = g.get("items") or []
            lines.append(f"\n【{g['scenario']}情景】")
            if group_items:
                lines.extend(_fmt_item(x) for x in group_items)
            else:
                lines.append("- 该情景下暂未检索到满足硬约束的房源")

        if similar_interest_items:
            lines.append("\n【基于您关注房源的相似推荐】")
            lines.extend(_fmt_item(x) for x in similar_interest_items)

        strict_final_text = "\n".join(lines)
        status = "scenario_split"
    elif validated:
        lines = []
        if mcp_interest_row:
            lines.append(
                f"我看到您在页面浏览了「{mcp_interest_row.get('title')}」，结合该房源为您推荐以下更匹配的选择："
            )
        elif user_interest_rows:
            user_titles = "、".join(str(x.get("title")) for x in user_interest_rows[:3])
            lines.append(f"我看到您在对话中提到「{user_titles}」，结合这些兴趣房源为您推荐：")
        else:
            lines.append("为您匹配到以下真实房源（仅展示工具结果且已做ID+价格校验）：")
        lines.extend(_fmt_item(x) for x in validated)
        if interest_row and similar_interest_items:
            lines.append("\n【基于您关注房源的相似推荐】")
            lines.extend(_fmt_item(x) for x in similar_interest_items)
        strict_final_text = "\n".join(lines)
        status = "success"
    elif similar_interest_items:
        lines = ["目前没有找到完全匹配您全部条件的房源。"]
        if mcp_interest_row:
            lines.append(f"我看到您在页面浏览了「{mcp_interest_row.get('title')}」，先给您推荐几套相似房源：")
        elif user_interest_rows:
            user_titles = "、".join(str(x.get("title")) for x in user_interest_rows[:3])
            lines.append(f"我看到您在对话中提到「{user_titles}」，先给您推荐几套相似房源：")
        lines.extend(_fmt_item(x) for x in similar_interest_items)
        strict_final_text = "\n".join(lines)
        status = "interest_fallback"
    elif fallback_items:
        lines = [
            "目前没有完全符合您条件的房源。",
            "为方便您先浏览，我从数据库中随机推荐了以下房源：",
        ]
        lines.extend(_fmt_item(x) for x in fallback_items)
        strict_final_text = "\n".join(lines)
        status = "fallback"
    else:
        strict_final_text = "目前没有符合条件的房源，且暂时没有可用的备选房源。"
        status = "incomplete"

    outcome = {
        "status": status,
        "final_text": strict_final_text,
        "strict_mode": True,
        "validated_ids": [str(x.get("id")) for x in validated],
        "fallback_ids": [str(x.get("id")) for x in fallback_items],
        "interest_property_id": str(interest_row.get("id")) if interest_row else None,
        "mcp_interest_property_id": str(mcp_interest_row.get("id")) if mcp_interest_row else None,
        "user_input_interest_property_ids": [str(x.get("id")) for x in user_interest_rows],
        "scenario_groups": [
            {
                "scenario": g.get("scenario"),
                "ids": [str(x.get("id")) for x in (g.get("items") or [])],
            }
            for g in scenario_results
        ],
        "llm_raw_text": final_ai.content if final_ai else "",
        "tool_observations": tool_observations,
    }

    return {
        "agent_outcomes": {
            **(state.get("agent_outcomes") or {}),
            "recommend": outcome,
        }
    }
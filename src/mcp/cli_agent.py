import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import redis
from langchain_core.messages import AIMessage, HumanMessage


def _bootstrap_paths() -> None:
    """确保直接执行 `python src/mcp/cli_agent.py` 时也能导入项目模块。"""
    current = Path(__file__).resolve()
    project_root = current.parents[2]  # house-agent/
    src_root = project_root / "src"

    for p in (str(project_root), str(src_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_graph():
    _bootstrap_paths()
    try:
        from agent import graph as app_graph
    except Exception:
        from src.agent import graph as app_graph
    return app_graph


def _read_web_state(redis_client: redis.Redis) -> dict[str, Any]:
    try:
        value = redis_client.get("current_user_intent")
        if value:
            return json.loads(value)
    except Exception:
        pass
    return {
        "business_context": {"viewing_property": "未知", "price": "未知"},
        "recent_action_trail": ["无端侧行为数据"],
    }


def _invoke_graph(
    app_graph,
    *,
    messages: list,
    user_id: str,
    web_state: dict[str, Any],
    need_reserve_decision: str = "不需要",
    reserve_form: dict[str, Any] | None = None,
) -> str:
    payload = {"messages": messages}
    context = {
        "user_id": user_id,
        "web_mcp_state": web_state,
        "interactive_mode": "cli",
        "need_reserve_decision": need_reserve_decision,
        "reserve_form": reserve_form,
    }
    try:
        result = app_graph.invoke(payload, context=context)
    except TypeError:
        result = app_graph.invoke(payload)
    except Exception as e:
        err = str(e)
        if "Model does not exist" in err or "model does not exist" in err:
            return (
                "模型调用失败：当前模型名在你的 OpenAI_API_BASE 下不可用。\n"
                "请在 .env 增加 OPENAI_MODEL，例如：\n"
                "OPENAI_MODEL=deepseek-ai/DeepSeek-V3\n"
                "然后重新 source .env 再启动 CLI。"
            )
        raise

    if isinstance(result, dict) and result.get("messages"):
        msg = result["messages"][-1]
        return getattr(msg, "content", str(msg))
    return str(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="House Agent CLI")
    parser.add_argument("--user-id", default="demo_user_001", help="用户ID（用于跨会话记忆）")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--disable-db", action="store_true", help="禁用推荐流程数据库依赖")
    args = parser.parse_args()

    if args.disable_db:
        os.environ["DISABLE_DB"] = "1"

    app_graph = _load_graph()
    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0, decode_responses=True)

    history: list = [AIMessage(content="你好，我是买房助手。输入 /help 查看命令。")]
    print("\n=== House Agent CLI 已启动 ===")
    print("命令：/help, /state, /reserve, /quit\n")

    while True:
        user_input = input("你> ").strip()
        if not user_input:
            continue

        if user_input in {"/quit", "/exit"}:
            print("已退出。")
            break

        if user_input == "/help":
            print("""
可用命令：
- 直接输入问题：正常对话调用主图
- /state：打印当前 Redis 端侧状态
- /reserve：快速预定（会提示输入标题/手机号/身份证）
- /quit：退出
""".strip())
            continue

        if user_input == "/state":
            print(json.dumps(_read_web_state(redis_client), ensure_ascii=False, indent=2))
            continue

        if user_input == "/reserve":
            title = input("房源标题> ").strip()
            phone = input("手机号> ").strip()
            id_card = input("身份证号> ").strip()

            reserve_form = {
                "title": title,
                "phone_number": phone,
                "id_card": id_card,
            }
            reserve_prompt = (
                f"我要预定房源：{title}。手机号：{phone}。身份证号：{id_card}。请直接帮我下单。"
            )
            web_state = _read_web_state(redis_client)
            history.append(HumanMessage(content=reserve_prompt))
            answer = _invoke_graph(
                app_graph,
                messages=history,
                user_id=args.user_id,
                web_state=web_state,
                need_reserve_decision="需要",
                reserve_form=reserve_form,
            )
            history.append(AIMessage(content=answer))
            print(f"助手> {answer}")
            continue

        web_state = _read_web_state(redis_client)
        history.append(HumanMessage(content=user_input))
        answer = _invoke_graph(
            app_graph,
            messages=history,
            user_id=args.user_id,
            web_state=web_state,
            need_reserve_decision="不需要",
        )
        history.append(AIMessage(content=answer))
        print(f"助手> {answer}")


if __name__ == "__main__":
    main()

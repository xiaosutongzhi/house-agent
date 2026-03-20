import asyncio
import json
import sys
import redis # 👈 引入 redis
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("mock-web-mcp")

# 初始化 Redis 客户端 (注意要和 Streamlit 连同一个库)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_purchase_intent_context",
            description="获取用户最近的前端行为轨迹(Action Trail)，用于辅助大模型判断购买意向",
            inputSchema={"type": "object", "properties": {}, "required": []}
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_purchase_intent_context":
        try:
            # 🚀 架构大跨越：从 Redis 中实时拉取！
            redis_data = redis_client.get("current_user_intent")
            
            if redis_data:
                # 原封不动地把 Redis 里的 JSON 字符串传给大模型
                return [TextContent(type="text", text=redis_data)]
            else:
                # 兜底：如果 Redis 是空的
                empty_state = {"business_context": "未浏览房源", "recent_action_trail": ["无近期行为"]}
                return [TextContent(type="text", text=json.dumps(empty_state, ensure_ascii=False))]
                
        except redis.ConnectionError:
            return [TextContent(type="text", text="系统错误：Redis 状态中心断开连接")]
            
    raise ValueError(f"找不到该工具: {name}")

async def main():
    sys.stderr.write("🚀 MCP Server (Redis 行为流水版) 已启动...\n")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
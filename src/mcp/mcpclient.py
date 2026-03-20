import asyncio
import json
from pathlib import Path
import sys

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: 'mcp'.\n"
        "Install it with: python -m pip install mcp\n"
        "Or run this target from the same activated venv/conda env you installed it in."
    ) from e

async def run_mcp_client():
    server_file = Path(__file__).with_name("mcpserver.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_file)], 
        env=None
    )

    print("Connecting to MCP stdio server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected.\n")

            print("Fetching purchase intent context from frontend...")
            # 注意这里工具名字变了
            result = await session.call_tool("get_purchase_intent_context", {})
            
            if result.content:
                frontend_data_str = result.content[0].text
                # 把字符串转回字典，方便格式化打印
                frontend_data = json.loads(frontend_data_str)

                business = frontend_data.get("business_context", {}) or {}
                trail = frontend_data.get("recent_action_trail", []) or []

                print("\nGot intent data:")
                print(f"Viewing: {business.get('viewing_property', 'unknown')}")
                print(f"Price: {business.get('price', 'unknown')}")
                print(f"Recent actions: {len(trail)}")
                if trail:
                    print(f"Latest action: {trail[-1]}")

if __name__ == "__main__":
    asyncio.run(run_mcp_client())
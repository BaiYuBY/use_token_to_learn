"""MCP with the official SDK, and MCP tools vs LangChain tools / 用官方 SDK 实现 MCP，以及 MCP 工具与 LangChain 工具的对比.

Run / 运行 (needs `pip install mcp==2.3.0 langgraph==1.2.19`):
    python docs/22-mcp-skills-sop/code/mcp_sdk.py

Covers / 包含:
    1. the same server in a few lines with the official Python SDK (MCPServer): the type hints and
       docstrings become the JSON schemas
       用官方 Python SDK（MCPServer）几行写出同一个服务器：类型注解和文档字符串变成 JSON schema
    2. interoperability: our hand-written client talks to the SDK server, and the SDK client talks to
       our hand-written server; that is the point of a protocol
       互通：手写的客户端连 SDK 服务器，SDK 客户端连手写的服务器；这正是“协议”的意义
    3. MCP tools inside LangChain / LangGraph: an adapter turns each MCP tool into a LangChain tool,
       and chapter 21's LangGraph agent runs on them unchanged
       在 LangChain / LangGraph 里用 MCP 工具：适配器把每个 MCP 工具变成 LangChain 工具，21 章的 LangGraph agent 原样运行
"""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "21-agents" / "code"))
from mcp_from_scratch import FINAL_ANSWER, StdioClient, calculate, mcp_to_loop_tools, make_caller, server_command  # noqa: E402

from agent_loop import ScriptedModel, agent_loop, make_tasks, make_world, run_tool  # noqa: E402

try:
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import StructuredTool
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters
    from mcp.server import MCPServer
except ImportError:
    raise SystemExit("This file needs the MCP SDK and LangGraph: pip install mcp==2.3.0 langgraph==1.2.19 / 需要先安装")

WORLD = make_world()

# ---------------------------------------------------------------------------
# 1. The server with the SDK / 用 SDK 写服务器
# ---------------------------------------------------------------------------
sdk_server = MCPServer("docs-sdk", instructions="Documents about made-up people and companies.")


@sdk_server.tool()
def search(query: str) -> str:
    """Search the documents. Returns the titles of the best matches."""
    return run_tool(WORLD, "search", {"query": query})


@sdk_server.tool()
def read(title: str) -> str:
    """Read the document with this exact title."""
    return run_tool(WORLD, "read", {"title": title})


@sdk_server.tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as 2026 - 1950."""
    return calculate(WORLD, expression)


@sdk_server.resource("doc://{title}")
def document(title: str) -> str:
    """One document, by title."""
    return WORLD["docs"].get(title.replace("%20", " "), "")


@sdk_server.prompt()
def answer_with_sources(question: str) -> str:
    """Answer a question from documents, citing them."""
    return f"Answer the question using only the documents and cite each title you used.\nQuestion: {question}"


# ---------------------------------------------------------------------------
# 3. MCP tools -> LangChain tools / MCP 工具 -> LangChain 工具
# ---------------------------------------------------------------------------
def mcp_to_langchain(client, mcp_tools):
    """Wrap each MCP tool as a LangChain StructuredTool. The JSON schema from tools/list becomes the
    LangChain tool's schema; calling it sends tools/call to the server.
    把每个 MCP 工具包装成 LangChain 的 StructuredTool：tools/list 给出的 JSON schema 就是工具的 schema，调用时向服务器发 tools/call。"""
    def wrap(tool):
        def run(**kwargs):
            text, _ = client.call_tool(tool["name"], kwargs)
            return text
        return StructuredTool.from_function(func=run, name=tool["name"], description=tool["description"], args_schema=tool["inputSchema"])
    return [wrap(t) for t in mcp_tools]


async def sdk_client_session(command):
    """The official client against any stdio server. 官方客户端连接任意 stdio 服务器。"""
    async with Client(StdioServerParameters(command=command[0], args=command[1:])) as client:
        tools = (await client.list_tools()).tools
        result = await client.call_tool("read", {"title": "Tafis Labs"})
        resource = await client.read_resource("doc://Tafis%20Labs")
        return [t.name for t in tools], result.content[0].text, resource.contents[0].text


def run_async(coro):
    """asyncio.run, also inside Jupyter (whose event loop is already running). 在 Jupyter 里也能用的 asyncio.run。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


def main():
    print("== 1. the server with the SDK / 用 SDK 写的服务器 ==")
    ours = StdioClient(server_command(), name="hand-written-client")
    sdk = StdioClient(server_command(__file__), name="hand-written-client")
    ours_tools, sdk_tools = ours.list_tools(), sdk.list_tools()
    print(f"  SDK server: {sdk.discover()['_meta']['io.modelcontextprotocol/serverInfo']}")
    print(f"  schema generated from `def read(title: str) -> str`: {sdk_tools[1]['inputSchema']}")
    for a, b in zip(ours_tools, sdk_tools):
        assert (a["name"], a["description"], a["inputSchema"]["required"]) == (b["name"], b["description"], b["inputSchema"]["required"])

    print("\n== 2. interoperability / 互通 ==")
    text, _ = sdk.call_tool("read", {"title": "Tafis Labs"})
    print(f"  hand-written client -> SDK server: read -> {text}")
    names, text2, res = run_async(sdk_client_session(server_command()))
    print(f"  SDK client -> hand-written server: tools {names}; read -> {text2}; resources/read -> {res[:30]}...")
    assert text == text2 == res
    world_tasks = make_tasks(WORLD, n=20)
    same = total = 0
    for client in [ours, sdk]:
        tools = mcp_to_loop_tools(client.list_tools()) + [FINAL_ANSWER]
        call = make_caller({t["name"]: (client, t["name"]) for t in tools if t["name"] != "final_answer"})
        for k, ts in world_tasks.items():
            for question, truth in ts:
                answer = agent_loop(ScriptedModel(WORLD), WORLD, question, tools=tools, call_tool=call)[0]
                same += answer == truth
                total += 1
    print(f"  the chapter 21 loop on either server: {same}/{total} correct")
    assert same == total

    print("\n== 3. MCP tools in LangGraph / 在 LangGraph 中使用 MCP 工具 ==")
    from langgraph_agent import build_agent, tool_trace
    lc_tools = mcp_to_langchain(sdk, sdk_tools)
    print(f"  {len(lc_tools)} LangChain tools made from tools/list; schema of read: {lc_tools[1].args}")
    app = build_agent(ScriptedModel(WORLD), tools=lc_tools)
    same = total = 0
    for k, ts in world_tasks.items():
        for question, truth in ts:
            out = app.invoke({"messages": [HumanMessage(question)]})
            hand = agent_loop(ScriptedModel(WORLD), WORLD, question)
            same += out["messages"][-1].content == truth and tool_trace(out["messages"]) == tool_trace(hand[3])
            total += 1
    print(f"  {same}/{total} tasks: the LangGraph agent with MCP-backed tools makes the same tool calls as chapter 21")
    assert same == total
    print("  a LangChain tool is a Python object inside one program; an MCP tool lives in a separate server that any")
    print("  MCP host (any language, any framework) can discover and call. The adapter joins the two.")
    ours.close()
    sdk.close()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        sdk_server.run()  # stdio transport / stdio 传输
    else:
        main()

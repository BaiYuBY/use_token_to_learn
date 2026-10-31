"""The Model Context Protocol from scratch / 从零实现 MCP.

Run / 运行:
    python docs/22-mcp-skills-sop/code/mcp_from_scratch.py

Covers / 包含:
    1. JSON-RPC 2.0 messages over stdio: requests, responses, errors
       stdio 上的 JSON-RPC 2.0 消息：请求、响应、错误
    2. a server with the three server features: tools, resources and prompts, plus server/discover
       一个服务器，提供三种能力：工具、资源和提示，以及 server/discover
    3. a client that starts the server as a subprocess and stamps every request with _meta
       (protocol revision 2026-07-28 is stateless: no initialize handshake)
       一个客户端：把服务器作为子进程启动，每个请求都带上 _meta（2026-07-28 版协议是无状态的，没有 initialize 握手）
    4. a host: chapter 21's agent loop with tools discovered from the server; it gives the same
       answers as with local tools
       宿主：21 章的 agent loop，工具从服务器发现而来；答案与使用本地工具时完全相同
    5. several servers at once: name clashes, and how tool descriptions pile up in every prompt
       同时连接多个服务器：重名，以及工具说明怎样在每次调用的提示里越堆越多

Only the standard library; no API key. The server here is the chapter 21 world of documents.
只用标准库，不需要 API key。服务器提供的是 21 章那个文档世界。
"""

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "21-agents" / "code"))
from agent_loop import TOOLS as LOCAL_TOOLS  # noqa: E402
from agent_loop import ScriptedModel, agent_loop, make_tasks, make_world, run_tool  # noqa: E402

PROTOCOL = "2026-07-28"
META_KEY = "io.modelcontextprotocol/"
# JSON-RPC error codes / JSON-RPC 错误码
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, UNSUPPORTED_VERSION = -32700, -32600, -32601, -32602, -32022


# ---------------------------------------------------------------------------
# 2. The server / 服务器
# ---------------------------------------------------------------------------
class DocsServer:
    """An MCP server for the documents of chapter 21. It only maps JSON-RPC requests to Python
    functions; the transport (stdio) is in serve_stdio below.
    21 章文档的 MCP 服务器。它只负责把 JSON-RPC 请求映射到 Python 函数；传输（stdio）在下面的 serve_stdio 里。"""

    name, version = "docs", "1.0"

    def __init__(self, world):
        self.world = world
        self.tools = {  # name -> (description, input schema, annotations, function)
            "search": ("Search the documents. Returns the titles of the best matches.",
                       {"query": "string"}, {"readOnlyHint": True}, lambda a: run_tool(world, "search", a)),
            "read": ("Read the document with this exact title.",
                     {"title": "string"}, {"readOnlyHint": True}, lambda a: run_tool(world, "read", a)),
            "calculator": ("Evaluate an arithmetic expression such as 2026 - 1950.",
                           {"expression": "string"}, {"readOnlyHint": True}, lambda a: calculate(world, a["expression"])),
        }
        self.methods = {
            "server/discover": self.discover,
            "tools/list": self.list_tools, "tools/call": self.call_tool,
            "resources/list": self.list_resources, "resources/read": self.read_resource,
            "prompts/list": self.list_prompts, "prompts/get": self.get_prompt,
        }

    # -- the protocol layer / 协议层 --
    def handle(self, msg):
        """One JSON-RPC message in, one response out (None for a notification). 一条消息进，一条响应出（通知没有响应）。"""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
            return error(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST, "Invalid Request")
        if "id" not in msg:
            return None  # a notification: nothing to answer / 通知：不用回复
        params = msg.get("params") or {}
        meta = params.get("_meta") or {}
        version = meta.get(META_KEY + "protocolVersion")
        if version is None:
            return error(msg["id"], INVALID_PARAMS, "every request must carry _meta with a protocolVersion")
        if version != PROTOCOL:
            return error(msg["id"], UNSUPPORTED_VERSION, "Unsupported protocol version",
                         {"requested": version, "supported": [PROTOCOL]})
        if msg["method"] not in self.methods:
            return error(msg["id"], METHOD_NOT_FOUND, f"Method not found: {msg['method']}")
        try:
            result = self.methods[msg["method"]]({k: v for k, v in params.items() if k != "_meta"})
        except ParamsError as e:
            return error(msg["id"], INVALID_PARAMS, str(e))
        result = {"resultType": "complete", **result, "_meta": {META_KEY + "serverInfo": {"name": self.name, "version": self.version}}}
        return {"jsonrpc": "2.0", "id": msg["id"], "result": result}

    # -- the features / 具体能力 --
    def discover(self, params):
        return {"supportedVersions": [PROTOCOL], "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "instructions": "Documents about made-up people and companies. Search, then read by exact title.",
                "ttlMs": 3_600_000, "cacheScope": "public"}

    def list_tools(self, params):
        tools = [{"name": n, "description": d,
                  "inputSchema": {"type": "object", "properties": {k: {"type": t} for k, t in s.items()}, "required": list(s)},
                  "annotations": ann} for n, (d, s, ann, _) in self.tools.items()]
        return {"tools": tools, "ttlMs": 3_600_000, "cacheScope": "public"}

    def call_tool(self, params):
        name, args = params.get("name"), params.get("arguments", {})
        if name not in self.tools:
            raise ParamsError(f"Unknown tool: {name}")
        _, spec, _, fn = self.tools[name]
        bad = [k for k in spec if not isinstance(args, dict) or not isinstance(args.get(k), str)]
        if bad:  # a tool-level error goes back to the model as a result with isError / 工具层面的错误作为结果返回给模型
            return {"content": [{"type": "text", "text": f"Invalid arguments: {bad} must be strings"}], "isError": True}
        text = fn(args)
        return {"content": [{"type": "text", "text": text}], "isError": text.startswith(("Error", "No document"))}

    def list_resources(self, params):
        docs = sorted(self.world["docs"])
        start = int(params.get("cursor") or 0)
        page = [{"uri": f"doc://{quote(t)}", "name": t, "mimeType": "text/plain"} for t in docs[start:start + 50]]
        out = {"resources": page, "ttlMs": 60_000, "cacheScope": "public"}
        if start + 50 < len(docs):
            out["nextCursor"] = str(start + 50)  # pagination / 分页
        return out

    def read_resource(self, params):
        uri = params.get("uri", "")
        title = unquote(uri.removeprefix("doc://"))
        if not uri.startswith("doc://") or title not in self.world["docs"]:
            raise ParamsError(f"Resource not found: {uri}")
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": self.world["docs"][title]}], "ttlMs": 60_000, "cacheScope": "public"}

    def list_prompts(self, params):
        return {"prompts": [{"name": "answer_with_sources", "description": "Answer a question from documents, citing them.",
                             "arguments": [{"name": "question", "required": True}]}], "ttlMs": 3_600_000, "cacheScope": "public"}

    def get_prompt(self, params):
        if params.get("name") != "answer_with_sources" or "question" not in (params.get("arguments") or {}):
            raise ParamsError("prompts/get needs name='answer_with_sources' and a question")
        q = params["arguments"]["question"]
        text = (f"Answer the question using only the documents. Search first, read the documents by exact title, "
                f"and cite each title you used in [brackets].\nQuestion: {q}")
        return {"description": "Answer with sources", "messages": [{"role": "user", "content": {"type": "text", "text": text}}]}


def calculate(world, expression):
    """'2026 - 1950' -> '2026 - 1950 = 76', so the result says what was computed. 结果里带上算式。"""
    value = run_tool(world, "calculator", {"expression": expression})
    return value if value.startswith("Error") else f"{expression} = {value}"


class ParamsError(Exception):
    pass


def error(id_, code, message, data=None):
    err = {"code": code, "message": message} | ({"data": data} if data is not None else {})
    return {"jsonrpc": "2.0", "id": id_, "error": err}


def quote(title):
    return title.replace(" ", "%20")


def unquote(s):
    return s.replace("%20", " ")


def serve_stdio(server):
    """The stdio transport: one JSON-RPC message per line on stdin, one per line on stdout.
    Logs go to stderr, never stdout. stdio 传输：stdin 上每行一条消息，stdout 上每行一条；日志写到 stderr，绝不能写到 stdout。"""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            reply = error(None, PARSE_ERROR, "Parse error")
        else:
            reply = server.handle(msg)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# 3. The client / 客户端
# ---------------------------------------------------------------------------
class StdioClient:
    """Starts a server command as a subprocess and talks JSON-RPC to it over stdin/stdout.
    把服务器命令作为子进程启动，通过 stdin/stdout 与它说 JSON-RPC。"""

    def __init__(self, command, name="client", trace=None):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.next_id = 0
        self.info = {"name": name, "version": "0.1"}
        self.trace = trace  # a list to record every message, for printing / 记录每条消息

    def request(self, method, params=None):
        self.next_id += 1
        meta = {META_KEY + "protocolVersion": PROTOCOL, META_KEY + "clientCapabilities": {}, META_KEY + "clientInfo": self.info}
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": {**(params or {}), "_meta": meta}}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline())
        if self.trace is not None:
            self.trace += [("→", msg), ("←", reply)]
        assert reply.get("id") == msg["id"]  # one request at a time here; real clients match ids / 真实客户端按 id 配对
        if "error" in reply:
            raise RuntimeError(f"{reply['error']['code']}: {reply['error']['message']}")
        return reply["result"]

    def send_raw(self, msg):
        """Send a message exactly as given (for showing errors). 原样发送一条消息（用来演示错误）。"""
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def discover(self):
        return self.request("server/discover")

    def list_tools(self):
        return self.request("tools/list")["tools"]

    def call_tool(self, name, arguments):
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        return "\n".join(c["text"] for c in result["content"] if c["type"] == "text"), result.get("isError", False)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def server_command(script=__file__, *args):
    return [sys.executable, str(script), "--serve", *args]


# ---------------------------------------------------------------------------
# 4. The host: chapter 21's loop with MCP tools / 宿主：用 MCP 工具的 21 章循环
# ---------------------------------------------------------------------------
FINAL_ANSWER = next(t for t in LOCAL_TOOLS if t["name"] == "final_answer")  # the host's own tool / 宿主自己的工具


def mcp_to_loop_tools(mcp_tools, prefix=""):
    """MCP tool descriptions -> the format the chapter 21 loop puts in the system prompt.
    MCP 的工具说明 -> 21 章循环放进系统提示的格式。"""
    return [{"name": prefix + t["name"], "description": t["description"], "parameters": t["inputSchema"]} for t in mcp_tools]


def make_caller(clients):
    """Route a tool name to the server that has it. 把工具名路由到提供它的服务器。"""
    def call(name, args):
        client, tool = clients[name]
        text, _ = client.call_tool(tool, args)
        return text
    return call


def words(tools):
    return len(json.dumps(tools).split())


def main():
    world = make_world()
    print("== 1-3. a session, message by message / 逐条消息看一次会话 ==")
    trace = []
    client = StdioClient(server_command(), name="demo-host", trace=trace)
    info = client.discover()
    tools = client.list_tools()
    text, is_error = client.call_tool("read", {"title": "Tafis Labs"})
    for arrow, m in trace:
        print(f"  {arrow} {json.dumps(m)[:150]}")
    assert info["supportedVersions"] == [PROTOCOL] and [t["name"] for t in tools] == ["search", "read", "calculator"]
    assert text.startswith("Tafis Labs was founded") and not is_error

    print("\n  resources, prompts and errors / 资源、提示和错误：")
    res = client.request("resources/list")
    print(f"  resources/list -> {len(res['resources'])} of {len(world['docs'])} documents, nextCursor={res.get('nextCursor')!r}")
    first = res["resources"][0]["uri"]
    print(f"  resources/read {first} -> {client.request('resources/read', {'uri': first})['contents'][0]['text'][:70]}")
    prompt = client.request("prompts/get", {"name": "answer_with_sources", "arguments": {"question": "Where does X work?"}})
    print(f"  prompts/get -> {prompt['messages'][0]['content']['text'][:80]}...")
    print(f"  tools/call with a bad argument -> isError={client.call_tool('read', {'title': 42})[1]}  (the model sees it and retries)")
    for method, params, code in [("tools/delete_everything", {}, METHOD_NOT_FOUND), ("resources/read", {"uri": "doc://Nobody"}, INVALID_PARAMS)]:
        try:
            client.request(method, params)
        except RuntimeError as e:
            print(f"  {method} -> error {e}")
            assert str(e).startswith(str(code))
    reply = client.send_raw({"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}})
    print(f"  a request without _meta -> {reply['error']['message']}")
    assert reply["error"]["code"] == INVALID_PARAMS
    client.close()

    print("\n== 4. the agent loop with tools from the server / 用服务器提供的工具运行 agent loop ==")
    client = StdioClient(server_command(), name="agent-host")
    loop_tools = mcp_to_loop_tools(client.list_tools()) + [FINAL_ANSWER]
    call = make_caller({t["name"]: (client, t["name"]) for t in loop_tools if t["name"] != "final_answer"})
    print(f"  system prompt tools: {[t['name'] for t in loop_tools]} ({words(loop_tools)} words; local version {words(LOCAL_TOOLS)})")
    tasks = make_tasks(world, n=30)
    same = total = 0
    for e in [0.0, 0.2]:
        for k, ts in tasks.items():
            local, remote = ScriptedModel(world, e, seed=k), ScriptedModel(world, e, seed=k)
            for question, truth in ts:
                a = agent_loop(local, world, question)
                b = agent_loop(remote, world, question, tools=loop_tools, call_tool=call)
                same += a[0] == b[0] and [m["content"] for m in a[3][1:]] == [m["content"] for m in b[3][1:]]
                total += 1
    print(f"  {same}/{total} runs (error rates 0% and 20%): same answer and same transcript as with local tools")
    assert same == total
    client.close()

    print("\n== 5. several servers / 多个服务器 ==")
    clients = {"docs": StdioClient(server_command(), "host"), "archive": StdioClient(server_command(), "host")}
    listed = {s: c.list_tools() for s, c in clients.items()}
    names = [t["name"] for ts in listed.values() for t in ts]
    clashes = sorted({n for n in names if names.count(n) > 1})
    print(f"  two servers both offer {clashes}: the host must namespace them, e.g. docs__read and archive__read")
    merged = [t for s, ts in listed.items() for t in mcp_to_loop_tools(ts, prefix=f"{s}__")] + [FINAL_ANSWER]
    assert len({t["name"] for t in merged}) == len(merged)
    per_tool = words(mcp_to_loop_tools(listed["docs"])) / len(listed["docs"])
    print(f"  each tool description is about {per_tool:.0f} words, re-sent on every model call:")
    for n_tools in [3, 30, 100, 300]:
        print(f"    {n_tools:3d} tools -> {n_tools * per_tool:6.0f} words per call, {6 * n_tools * per_tool:7.0f} words for a 6-call task")
    for c in clients.values():
        c.close()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve_stdio(DocsServer(make_world()))
    else:
        main()

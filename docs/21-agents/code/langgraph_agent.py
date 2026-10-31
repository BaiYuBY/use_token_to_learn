"""The same agent with LangChain and LangGraph / 用 LangChain 和 LangGraph 实现同一个 agent.

Run / 运行 (needs `pip install langgraph`, which brings langchain-core):
    python docs/21-agents/code/langgraph_agent.py

Covers / 包含:
    1. LangChain tools: a typed Python function with a docstring becomes a JSON-schema tool
       LangChain 工具：带类型和文档字符串的 Python 函数自动变成 JSON schema 工具
    2. the agent loop as a LangGraph StateGraph: an "agent" node, a "tools" node, a conditional edge,
       plus a "check" node; it takes exactly the same steps as the hand-written loop
       把 agent loop 写成 LangGraph 的状态图：agent 节点、tools 节点、条件边，再加一个 check 节点；
       它走的每一步都和手写循环相同
    3. checkpoints: every step's state is saved and can be inspected or resumed
       检查点：每一步的状态都被保存，可以查看、可以从中断处继续
    4. human in the loop: pause before a risky tool and wait for approval
       人工介入：执行有风险的工具前暂停，等人批准
    5. LCEL: composing runnables with |, as a retrieval chain from chapter 19
       LCEL：用 | 组合可运行对象，例如 19 章的检索链
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_loop import ScriptedModel, agent_loop, make_tasks, make_world, run_tool, supported  # noqa: E402

try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import tool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode
    from langgraph.types import Command, interrupt
except ImportError:
    raise SystemExit("This file needs LangGraph: pip install langgraph==1.2.19 / 需要先安装 LangGraph")

WORLD = make_world()


# ---------------------------------------------------------------------------
# 1. Tools / 工具
# ---------------------------------------------------------------------------
@tool
def search(query: str) -> str:
    """Search the documents. Returns the titles of the best matches."""
    return run_tool(WORLD, "search", {"query": query})


@tool
def read(title: str) -> str:
    """Read the document with this exact title."""
    return run_tool(WORLD, "read", {"title": title})


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as 2026 - 1950."""
    return f"{expression} = {run_tool(WORLD, 'calculator', {'expression': expression})}"


TOOLS = [search, read, calculator]


# ---------------------------------------------------------------------------
# 2. The agent as a state graph / 状态图形式的 agent
# ---------------------------------------------------------------------------
def to_plain(messages):
    """LangChain messages -> the plain dicts the scripted model reads. LangChain 消息 -> 脚本模型读取的字典。"""
    out = [{"role": "system", "content": ""}]
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content})
        else:
            out.append({"role": "assistant", "content": m.content})
    return out


def build_agent(model, check=True, checkpointer=None):
    """START -> agent -> (tool calls? -> tools -> agent) -> (final answer -> check -> END or back to agent).
    The prebuilt ToolNode runs the tools; the routing functions are the loop's if-statements.
    预置的 ToolNode 负责执行工具；路由函数就是循环里的 if 判断。"""
    counter = {"n": 0}

    def agent(state: MessagesState):
        calls = model(to_plain(state["messages"]))
        tool_calls = [c for c in calls if c["name"] != "final_answer"]
        counter["n"] += 1
        if tool_calls:
            # a malformed call (arguments not an object) goes through with no arguments; ToolNode then
            # returns the validation error to the model, as the hand-written loop does
            # 格式错误的调用（参数不是对象）以空参数传过去；ToolNode 会把校验错误返回给模型，和手写循环一样
            return {"messages": [AIMessage(content="", tool_calls=[
                {"name": c["name"], "args": c["args"] if isinstance(c["args"], dict) else {}, "id": f"call_{counter['n']}_{i}"}
                for i, c in enumerate(tool_calls)])]}
        return {"messages": [AIMessage(content=calls[0]["args"]["answer"])]}

    def route_after_agent(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if last.tool_calls else ("check" if check else END)

    def check_answer(state: MessagesState):
        answer = state["messages"][-1].content
        if supported(answer, to_plain(state["messages"][:-1])):
            return {}
        return {"messages": [HumanMessage("Check failed: that answer is not in any tool result. Keep working.")]}

    def route_after_check(state: MessagesState):
        return "agent" if isinstance(state["messages"][-1], HumanMessage) else END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("check", check_answer)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "check", END])
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("check", route_after_check, ["agent", END])
    return graph.compile(checkpointer=checkpointer)


def tool_trace(messages):
    """The sequence of tool calls, for comparing two runs. 工具调用序列，用来比较两次运行。"""
    out = []
    for m in messages:
        if isinstance(m, AIMessage):
            out += [(c["name"], tuple(sorted(c["args"].items()))) for c in m.tool_calls]
        elif isinstance(m, dict) and m["role"] == "assistant":
            out += [(c["name"], tuple(sorted(c["args"].items()))) for c in json.loads(m["content"]) if c["name"] != "final_answer"]
    return out


# ---------------------------------------------------------------------------
# 4. Human in the loop / 人工介入
# ---------------------------------------------------------------------------
def build_payment_agent(checkpointer):
    """A two-node graph: the agent wants to send money; the guard node pauses with interrupt() and waits
    for a person's decision, which arrives later as Command(resume=...).
    两个节点的图：agent 想转账；guard 节点用 interrupt() 暂停，等人的决定，之后通过 Command(resume=...) 传回来。"""

    def agent(state: MessagesState):
        last = state["messages"][-1]
        if isinstance(last, ToolMessage):
            return {"messages": [AIMessage(content=f"Done: {last.content}")]}
        return {"messages": [AIMessage(content="", tool_calls=[{"name": "transfer", "args": {"to": "Gasel Bank", "amount": 900}, "id": "t1"}])]}

    def guard(state: MessagesState):
        call = state["messages"][-1].tool_calls[0]
        decision = interrupt({"tool": call["name"], "args": call["args"], "question": "approve?"})
        result = f"sent {call['args']['amount']} to {call['args']['to']}" if decision == "yes" else "the user declined the transfer"
        return {"messages": [ToolMessage(content=result, tool_call_id=call["id"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("guard", guard)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", lambda s: "guard" if s["messages"][-1].tool_calls else END, ["guard", END])
    graph.add_edge("guard", "agent")
    return graph.compile(checkpointer=checkpointer)


def main():
    print("== 1. LangChain tools / LangChain 工具 ==")
    schema = convert_to_openai_tool(read)
    print(f"  @tool read(title: str) -> {schema['function']}")
    assert schema["function"]["parameters"]["required"] == ["title"]

    print("\n== 2. the same loop as a LangGraph state graph / 同一个循环，写成 LangGraph 状态图 ==")
    app = build_agent(ScriptedModel(WORLD))
    print("  " + app.get_graph().draw_mermaid().split("graph TD;")[1].strip().replace("\n", "\n  ")[:600])
    tasks = make_tasks(WORLD, n=30)
    same = 0
    total = 0
    for k, ts in tasks.items():
        for question, truth in ts:
            out = app.invoke({"messages": [HumanMessage(question)]})
            hand = agent_loop(ScriptedModel(WORLD), WORLD, question)
            same += out["messages"][-1].content == truth == hand[0] and tool_trace(out["messages"]) == tool_trace(hand[3])
            total += 1
    print(f"  {same}/{total} tasks: same answer and same tool calls as the hand-written loop")
    assert same == total

    print("\n== 3. checkpoints / 检查点 ==")
    saver = InMemorySaver()
    app = build_agent(ScriptedModel(WORLD, error_rate=0.3, seed=3), checkpointer=saver)
    config = {"configurable": {"thread_id": "demo"}}
    question, truth = tasks[5][0]
    out = app.invoke({"messages": [HumanMessage(question)]}, config)
    history = list(app.get_state_history(config))
    print(f"  {len(history)} checkpoints saved for one run; the latest has {len(history[0].values['messages'])} messages; "
          f"answer {out['messages'][-1].content!r} (truth {truth!r})")
    for snap in reversed(history[:-1]):
        if snap.next:
            print(f"    step {snap.metadata['step']:2d}: next node {snap.next[0]:6s} | messages so far {len(snap.values['messages'])}")
    assert len(history) > 5

    print("\n== 4. human in the loop / 人工介入 ==")
    for decision in ["yes", "no"]:
        app = build_payment_agent(InMemorySaver())
        config = {"configurable": {"thread_id": decision}}
        paused = app.invoke({"messages": [HumanMessage("Pay the invoice from Gasel Bank.")]}, config)
        request = paused["__interrupt__"][0].value
        out = app.invoke(Command(resume=decision), config)
        print(f"  paused before {request['tool']}({request['args']}); the person says {decision!r} -> {out['messages'][-1].content}")
        assert ("sent 900" in out["messages"][-1].content) == (decision == "yes")

    print("\n== 5. LCEL: a retrieval chain with | / LCEL：用 | 组成检索链 ==")
    retrieve = RunnableLambda(lambda q: search.invoke(q).removeprefix("Titles: ").split("; ")[0])
    fetch = RunnableLambda(lambda title: read.invoke(title))
    to_prompt = RunnableLambda(lambda doc: f"<source>{doc}</source>\nAnswer from the source.")
    chain = retrieve | fetch | to_prompt
    p = next(iter(WORLD["people"]))
    print("  " + chain.invoke(f"Where does {p} work?").replace("\n", "\n  "))
    print(f"  batch of 3 questions -> {len(chain.batch([f'Where does {q} work?' for q in list(WORLD['people'])[:3]]))} prompts")


if __name__ == "__main__":
    main()

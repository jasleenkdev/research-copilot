"""A hand-written tool-calling loop.

CONCEPT: bind_tools
`model.bind_tools(tools)` returns a new Runnable that attaches the tools' schemas
to every request. It doesn't run anything. It just tells the model which tools
exist, so the model can reply with `tool_calls` instead of, or alongside, text.

CONCEPT: The tool-call loop
    you    -> [SystemMessage, HumanMessage]
    model  -> AIMessage(tool_calls=[{name, args, id}])     "please run this"
    you    -> run it, append ToolMessage(result, tool_call_id=id)
    model  -> more tool_calls, or a final answer with no tool_calls
    ...repeat until an AIMessage has no tool_calls

The model keeps no state between calls. Each iteration resends the entire
message list, and that list is the agent's only memory. Phase 3 rebuilds this
same loop as a LangGraph StateGraph: `messages` becomes the graph's State, the
model call and tool execution become nodes, and the `if not tool_calls` check
becomes a conditional edge. Keep this file in mind when you get there.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool

from research_copilot.models import get_chat_model
from research_copilot.prompts import AGENT_PROMPT
from research_copilot.tools import search_arxiv


@dataclass
class AgentResult:
    answer: str
    messages: list[BaseMessage]
    iterations: int
    tool_calls_made: int
    # True when max_iterations ran out while the model still wanted tools.
    stopped_early: bool = False


def run_tool_loop(
    question: str,
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    max_iterations: int = 6,
    on_event: Callable[[str], None] | None = None,
) -> AgentResult:
    tools = list(tools) if tools is not None else [search_arxiv]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = (model or get_chat_model()).bind_tools(tools)
    log = on_event or (lambda _event: None)

    # format_messages() renders the template into concrete message objects. The
    # loop then works on that plain list, which grows as the loop runs.
    messages: list[BaseMessage] = AGENT_PROMPT.format_messages(question=question)
    tool_calls_made = 0

    for iteration in range(1, max_iterations + 1):
        ai_message = model_with_tools.invoke(messages)

        # Append the whole AIMessage, not just its text. It contains the
        # tool-call blocks, and the API rejects a tool result whose matching call
        # isn't in the history. With Claude it also contains thinking blocks,
        # which must be sent back unchanged.
        messages.append(ai_message)

        if ai_message.text and ai_message.tool_calls:
            log(f"[model] {ai_message.text}")

        if not ai_message.tool_calls:
            return AgentResult(
                answer=ai_message.text,
                messages=messages,
                iterations=iteration,
                tool_calls_made=tool_calls_made,
            )

        # One AIMessage can request several tools at once (parallel tool calls).
        # Each call gets its own ToolMessage, and all of them are appended before
        # the next model call. ChatAnthropic merges consecutive ToolMessages into
        # the single user turn the API expects.
        for call in ai_message.tool_calls:
            log(f"[tool] {call['name']}({call['args']})")
            messages.append(_execute_tool_call(call, tools_by_name))
            tool_calls_made += 1

    last_text = next(
        (m.text for m in reversed(messages) if isinstance(m, AIMessage) and m.text),
        "",
    )
    return AgentResult(
        answer=last_text or "(Stopped: hit max_iterations before a final answer.)",
        messages=messages,
        iterations=max_iterations,
        tool_calls_made=tool_calls_made,
        stopped_early=True,
    )


def _execute_tool_call(call: ToolCall, tools_by_name: dict[str, BaseTool]) -> ToolMessage:
    tool = tools_by_name.get(call["name"])
    if tool is None:
        return ToolMessage(
            content=f"Error: no tool named {call['name']!r}. Available: {sorted(tools_by_name)}",
            tool_call_id=call["id"],
            status="error",
        )
    try:
        # Passing the whole ToolCall dict, not just call["args"], makes the tool
        # return a ToolMessage with tool_call_id already filled in.
        return tool.invoke(call)
    except Exception as exc:
        # Bad arguments (schema validation) or a bug in the tool. Tell the model
        # so it can correct itself, instead of discarding the run so far.
        return ToolMessage(
            content=f"Error running {call['name']}: {exc}",
            tool_call_id=call["id"],
            status="error",
        )

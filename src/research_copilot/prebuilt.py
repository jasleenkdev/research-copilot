"""The same agent in four lines, using langgraph.prebuilt.create_react_agent.

Compare this file's length with graph.py's. They produce nearly the same graph.

CONCEPT: create_react_agent
It builds and compiles a StateGraph for the single most common agent shape - a
model with tools, looping until the model stops asking for them ("ReAct":
reason, act, observe, repeat). You hand it a model and a list of tools; it
returns a compiled graph.

WHAT IT DOES FOR YOU THAT graph.py DOES BY HAND
  - State schema. It defines `AgentState` (a `messages` key with the
    `add_messages` reducer, plus `remaining_steps`) so you never write a
    TypedDict or choose a reducer. state.py exists only because we wanted to.
  - bind_tools. It calls `model.bind_tools(tools)` for you.
  - The tool node. `ToolNode` runs every requested tool, in parallel when there
    are several, and turns each result into a ToolMessage with the right
    `tool_call_id`. It also catches tool exceptions and returns them to the
    model as error ToolMessages - the same policy `_execute_tool_call` spells
    out in agent_loop.py.
  - The routing. Its built-in `tools_condition` is our `should_continue`:
    tool_calls on the last message -> the tool node, otherwise END.
  - The edges and compile(). Entry at the model, model -> tools, tools -> model,
    model -> END. Same cycle, wired for you.
  - Message accumulation. Falls out of the state schema it chose.
  - A recursion guard via `remaining_steps`, so the loop can't run forever.

WHAT IT DOES *NOT* DO, AND WHY PHASE 3 IS HAND-ROLLED ANYWAY
  - Extra state. Its state is a transcript. Our `mode`, `documents`, `context`,
    and `iterations` need a custom `state_schema`, and from Phase 4 on the state
    is the point: `research_notes`, `draft`, `critique`, `iteration_count`.
  - Extra nodes and branching. There is no room in it for `retrieve_docs`, or
    for a router that picks a strategy before the model runs. It is one model,
    one tool node, one loop.
  - Anything that isn't the ReAct shape. The reflection loop of Phase 5
    (draft -> critique -> revise) and the Supervisor of Phase 6 are different
    graph shapes, not configurations of this one.

THE TRADE-OFF, PLAINLY
Reach for `create_react_agent` when the shape genuinely is "model plus tools,
loop until done" - it is less code to write, to read, and to get wrong. Write
the graph out when you need state beyond a transcript, a node that isn't a model
or a tool, or a branch the prebuilt doesn't have. The honest reason to hand-roll
it once, here, is that the prebuilt is opaque until you have built the thing it
is hiding: when a prebuilt agent loops when it shouldn't, or drops a message you
expected to survive, "it's a StateGraph with a conditional edge on tool_calls"
is the sentence that lets you debug it.

`create_react_agent` is not a toy, and it is not a different framework - it
returns the same `CompiledStateGraph` that `builder.compile()` returns, with the
same `.invoke()`, the same `.get_graph().draw_mermaid()`, and the same
checkpointer support. You can always print its graph and read what it built.

NOTE: this import is deprecated as of LangGraph 1.0
`langgraph.prebuilt.create_react_agent` has moved to `langchain.agents`:

    from langchain.agents import create_agent
    agent = create_agent(model=..., tools=[...], system_prompt="...")

Running this file prints a LangGraphDeprecatedSinceV10 warning, and the old name
is slated for removal in LangGraph 2.0. It is kept here because the phase plan
names it, and because the lesson is about what a prebuilt hides, which is
identical either way - `create_agent` builds the same model/tool loop, renames
`prompt` to `system_prompt`, and adds a `middleware` hook for injecting
behaviour around the model call. Switch the two lines in `build_prebuilt_agent`
whenever you like; nothing else in the project depends on which one is used.
"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from research_copilot.models import get_chat_model
from research_copilot.prompts import AGENT_SYSTEM_PROMPT
from research_copilot.tools import search_arxiv


def build_prebuilt_agent(
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
) -> Runnable:
    """graph.py's live-search path, in one call.

    `prompt` is the prebuilt's name for the system message. Passing a plain
    string makes it a SystemMessage prepended to every model call - exactly what
    `call_model` does by hand in graph.py.
    """
    return create_react_agent(
        model=model or get_chat_model(),
        tools=list(tools) if tools is not None else [search_arxiv],
        prompt=AGENT_SYSTEM_PROMPT,
        name="research-copilot-prebuilt",
    )


def run_prebuilt_agent(question: str, *, agent: Runnable | None = None) -> dict:
    """Invoke it. The input and output shapes match graph.py's, minus our extra keys.

    Note what the input does *not* carry: no `mode`, no `question`. The
    prebuilt's state is the transcript and nothing else, so the question can only
    enter as a message.
    """
    agent = agent or build_prebuilt_agent()
    return agent.invoke({"messages": [HumanMessage(content=question)]})


def make_graph(config: dict | None = None) -> Runnable:
    """Factory for the LangGraph dev server / Studio (see langgraph.json).

    Studio lists both graphs side by side, which is the quickest way to see the
    structural difference: this one has two nodes, graph.py's has three and an
    entry branch.
    """
    return build_prebuilt_agent()

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from research_copilot.agent_loop import run_tool_loop


class FakeToolCallingModel(FakeMessagesListChatModel):
    """Replays scripted AIMessages. bind_tools is a no-op because the script already
    decides when tools get called."""

    def bind_tools(self, tools, **kwargs):
        return self


@tool
def search_arxiv(query: str) -> str:
    """Stub search."""
    return f"[1] A paper about {query}"


def tool_call(name, args, call_id):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def test_loop_runs_tool_then_returns_final_answer():
    model = FakeToolCallingModel(
        responses=[
            tool_call("search_arxiv", {"query": "rag"}, "call_1"),
            AIMessage(content="Final answer citing [1]."),
        ]
    )
    result = run_tool_loop("What is RAG?", model=model, tools=[search_arxiv])

    assert result.answer == "Final answer citing [1]."
    assert result.iterations == 2
    assert result.tool_calls_made == 1
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert tool_messages[0].tool_call_id == "call_1"
    assert "A paper about rag" in tool_messages[0].content


def test_unknown_tool_is_reported_to_model_not_raised():
    model = FakeToolCallingModel(
        responses=[tool_call("web_search", {"q": "x"}, "call_1"), AIMessage(content="ok")]
    )
    result = run_tool_loop("Q", model=model, tools=[search_arxiv])
    error = next(m for m in result.messages if isinstance(m, ToolMessage))
    assert error.status == "error"
    assert "no tool named 'web_search'" in error.content
    assert result.answer == "ok"


def test_invalid_tool_args_are_reported_to_model():
    model = FakeToolCallingModel(
        responses=[tool_call("search_arxiv", {"wrong_arg": 1}, "call_1"), AIMessage(content="ok")]
    )
    result = run_tool_loop("Q", model=model, tools=[search_arxiv])
    error = next(m for m in result.messages if isinstance(m, ToolMessage))
    assert error.status == "error"


def test_max_iterations_stops_a_model_that_never_finishes():
    model = FakeToolCallingModel(responses=[tool_call("search_arxiv", {"query": "rag"}, "c")])
    result = run_tool_loop("Q", model=model, tools=[search_arxiv], max_iterations=3)
    assert result.stopped_early
    assert result.tool_calls_made == 3

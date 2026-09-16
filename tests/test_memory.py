from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from research_copilot.memory import ConversationMemory, render_messages
from research_copilot.prompts import CHAT_PROMPT


def build_history(turns: int, filler: str = "short") -> list:
    messages = []
    for i in range(turns):
        messages.append(HumanMessage(content=f"question {i} {filler}"))
        messages.append(AIMessage(content=f"answer {i} {filler}"))
    return messages


def test_trim_keeps_newest_and_starts_on_human():
    memory = ConversationMemory(strategy="trim", max_tokens=40)
    memory.messages = build_history(10)
    report = memory.prune()

    assert report.messages_after < report.messages_before
    assert memory.token_count() <= 40
    # The surviving history must begin with a human turn: the API expects the
    # first message after the system prompt to be the user's.
    assert isinstance(memory.messages[0], HumanMessage)
    # "Keep the newest" means the last exchange is still there.
    assert "question 9" in memory.messages[0].text or "answer 9" in memory.messages[-1].text


def test_budget_is_tokens_not_message_count():
    """Two histories with the same message count prune very differently."""
    short_memory = ConversationMemory(strategy="trim", max_tokens=200)
    short_memory.messages = build_history(4, filler="ok")

    long_memory = ConversationMemory(strategy="trim", max_tokens=200)
    long_memory.messages = build_history(4, filler="word " * 200)

    assert len(short_memory.messages) == len(long_memory.messages)
    short_memory.prune()
    long_memory.prune()
    assert len(short_memory.messages) == 8  # everything fits
    assert len(long_memory.messages) < 8  # same message count, far more tokens


def test_summarize_replaces_old_turns_with_a_summary_message():
    model = FakeListChatModel(responses=["User asked about RAG; assistant explained retrieval."])
    memory = ConversationMemory(
        strategy="summarize", max_tokens=40, keep_last_messages=2, summary_model=model
    )
    memory.messages = build_history(6)
    report = memory.prune()

    assert report.summarized
    assert memory.summary == "User asked about RAG; assistant explained retrieval."
    assert len(memory.messages) == 2
    assert isinstance(memory.messages[0], HumanMessage)

    variables = memory.prompt_variables()
    summary_message = variables["earlier_summary"][0]
    assert isinstance(summary_message, SystemMessage)
    assert "RAG" in summary_message.content


def test_summarize_folds_previous_summary_back_in():
    model = FakeListChatModel(responses=["combined summary"])
    memory = ConversationMemory(
        strategy="summarize", max_tokens=10, keep_last_messages=2, summary_model=model
    )
    memory.summary = "earlier summary"
    memory.messages = build_history(4)
    memory.prune()
    assert memory.summary == "combined summary"


def test_no_summary_means_no_placeholder_variable():
    memory = ConversationMemory(strategy="trim")
    memory.add_user_turn("hi")
    assert "earlier_summary" not in memory.prompt_variables()


def test_chat_prompt_renders_with_and_without_summary():
    memory = ConversationMemory(strategy="trim")
    memory.add_user_turn("what is RAG?")

    without = CHAT_PROMPT.format_messages(**memory.prompt_variables())
    assert [m.type for m in without] == ["system", "human"]

    memory.summary = "earlier stuff"
    with_summary = CHAT_PROMPT.format_messages(**memory.prompt_variables())
    assert [m.type for m in with_summary] == ["system", "system", "human"]


def test_prune_is_a_no_op_when_history_fits():
    memory = ConversationMemory(strategy="trim", max_tokens=5000)
    memory.messages = build_history(2)
    report = memory.prune()
    assert report.messages_before == report.messages_after == 4


def test_render_messages_labels_roles():
    text = render_messages([HumanMessage(content="q"), AIMessage(content="a")])
    assert text == "User: q\nAssistant: a"

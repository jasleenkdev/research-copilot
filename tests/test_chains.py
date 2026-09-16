import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_copilot.chains import build_answer_chain, build_structured_chain
from research_copilot.prompts import ANSWER_PROMPT
from research_copilot.schemas import ResearchAnswer


def test_answer_prompt_renders_system_then_human():
    messages = ANSWER_PROMPT.format_messages(question="What is RAG?")
    assert [type(m) for m in messages] == [SystemMessage, HumanMessage]
    assert messages[1].content == "What is RAG?"


def test_answer_chain_returns_plain_string():
    model = FakeListChatModel(responses=["RAG pairs retrieval with generation."])
    result = build_answer_chain(model).invoke({"question": "What is RAG?"})
    assert result == "RAG pairs retrieval with generation."


def test_structured_chain_parses_fenced_json():
    reply = """```json
{
  "question": "What is RAG?",
  "summary": "RAG retrieves documents and conditions generation on them.",
  "key_points": ["Retriever", "Generator"],
  "confidence": "high"
}
```"""
    model = FakeListChatModel(responses=[reply])
    result = build_structured_chain(model).invoke({"question": "What is RAG?", "notes": ""})
    assert isinstance(result, ResearchAnswer)
    assert result.confidence == "high"
    assert result.sources == []


def test_structured_chain_rejects_non_conforming_output():
    model = FakeListChatModel(responses=['{"summary": "missing fields", "confidence": "very"}'])
    with pytest.raises(OutputParserException):
        build_structured_chain(model).invoke({"question": "Q", "notes": ""})


def test_structured_prompt_contains_schema():
    chain = build_structured_chain(FakeListChatModel(responses=["unused"]))
    prompt = chain.first
    rendered = prompt.format_messages(question="Q", notes="")
    assert "key_points" in rendered[0].content

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from research_copilot.retrieval import build_rag_chain, describe_source, format_docs


def test_format_docs_numbers_excerpts_for_citation():
    docs = [
        Document(page_content="first chunk", metadata={"source": "a.txt"}),
        Document(page_content="second chunk", metadata={"source": "b.pdf", "page": 2}),
    ]
    text = format_docs(docs)
    assert "[1] (a.txt)" in text
    assert "[2] (b.pdf, page 3)" in text  # page metadata is 0-indexed


def test_format_docs_handles_no_results():
    assert format_docs([]) == "(no excerpts retrieved)"


def test_describe_source_without_page():
    assert describe_source(Document(page_content="x", metadata={"source": "s.md"})) == "s.md"


def test_rag_chain_returns_answer_and_the_docs_behind_it():
    docs = [Document(page_content="RAG retrieves then generates.", metadata={"source": "kb.md"})]
    retriever = RunnableLambda(lambda _question: docs)
    model = FakeListChatModel(responses=["RAG retrieves then generates [1]."])

    result = build_rag_chain(model=model, retriever=retriever).invoke("What is RAG?")

    assert result["question"] == "What is RAG?"
    assert result["docs"] == docs
    assert "[1] (kb.md)" in result["context"]
    assert result["answer"] == "RAG retrieves then generates [1]."


def test_rag_chain_runs_over_a_real_vector_store():
    """End-to-end with a real Chroma store, fake embeddings, and a fake model."""
    store = Chroma(
        collection_name="rag-test", embedding_function=DeterministicFakeEmbedding(size=32)
    )
    store.add_documents(
        [
            Document(page_content="Chunking splits documents.", metadata={"source": "a.md"}),
            Document(page_content="Embeddings map text to vectors.", metadata={"source": "b.md"}),
        ]
    )
    model = FakeListChatModel(responses=["answer"])
    chain = build_rag_chain(model=model, retriever=store.as_retriever(search_kwargs={"k": 2}))

    result = chain.invoke("how does chunking work?")
    assert len(result["docs"]) == 2
    assert result["answer"] == "answer"

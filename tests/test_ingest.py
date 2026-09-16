import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from research_copilot.ingest import chunk_documents, ingest_path, load_documents


@pytest.fixture
def docs_dir(tmp_path):
    (tmp_path / "notes.md").write_text("# Notes\n\n" + "alpha beta gamma. " * 40, encoding="utf-8")
    (tmp_path / "paper.txt").write_text("delta epsilon. " * 40, encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")  # unsupported
    return tmp_path


@pytest.fixture
def store():
    # In-memory Chroma with fake embeddings: no model download, no network, and
    # deterministic vectors, so these tests are fast and free.
    return Chroma(
        collection_name="test",
        embedding_function=DeterministicFakeEmbedding(size=32),
    )


def test_load_documents_walks_directory_and_skips_unsupported(docs_dir):
    documents, report = load_documents(docs_dir)
    assert len(report.files) == 2
    assert any(name.endswith("data.csv") for name in report.skipped)
    assert all("source" in d.metadata for d in documents)


def test_load_documents_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope.txt")


def test_chunking_respects_size_and_carries_metadata():
    text = "sentence one. " * 200
    chunks = chunk_documents(
        [Document(page_content=text, metadata={"source": "a.txt"})],
        chunk_size=200,
        chunk_overlap=40,
    )
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["source"] == "a.txt" for c in chunks)
    assert [c.metadata["chunk"] for c in chunks] == list(range(len(chunks)))
    assert all("start_index" in c.metadata for c in chunks)


def test_overlap_repeats_text_across_the_boundary():
    words = " ".join(f"w{i}" for i in range(200))
    no_overlap = chunk_documents(
        [Document(page_content=words, metadata={"source": "a"})],
        chunk_size=120,
        chunk_overlap=0,
    )
    overlapped = chunk_documents(
        [Document(page_content=words, metadata={"source": "a"})],
        chunk_size=120,
        chunk_overlap=60,
    )
    # Overlap buys redundancy: more chunks covering the same text.
    assert len(overlapped) > len(no_overlap)


def test_ingest_stores_chunks(docs_dir, store):
    report = ingest_path(docs_dir, store=store, chunk_size=200, chunk_overlap=20)
    assert report.chunks > 0
    assert report.total_in_store == report.chunks


def test_reingesting_replaces_instead_of_duplicating(docs_dir, store):
    first = ingest_path(docs_dir, store=store, chunk_size=200, chunk_overlap=20)
    second = ingest_path(docs_dir, store=store, chunk_size=200, chunk_overlap=20)
    assert second.total_in_store == first.total_in_store


def test_ingest_of_empty_directory_reports_nothing(tmp_path, store):
    report = ingest_path(tmp_path, store=store)
    assert report.files == []
    assert report.chunks == 0


def test_pdf_files_are_loaded_page_by_page(tmp_path, monkeypatch):
    """PDFs take the pypdf branch and get one Document per page, with page metadata."""

    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage("page one text"), FakePage("page two text")]

    monkeypatch.setattr("research_copilot.ingest.PdfReader", FakeReader)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    documents, report = load_documents(pdf_path)
    assert len(documents) == 2
    assert documents[1].metadata["page"] == 1
    assert documents[0].page_content == "page one text"
    assert report.files == [str(pdf_path)]

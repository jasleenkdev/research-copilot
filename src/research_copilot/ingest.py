"""Ingestion: load -> chunk -> embed -> store.

This is the offline half of RAG. It runs once per document set; retrieval.py
then queries what it produced.

CONCEPT: why chunk at all?
Two reasons, and they pull in the same direction:
  1. Retrieval precision. Embedding a whole 40-page PDF as one vector gives you
     one blurry average of everything it says, and it would match almost any
     question weakly and none of them well.
  2. Prompt budget. You paste retrieved text into the prompt, so the unit of
     retrieval has to be small enough that several of them fit.
Chunking is the main quality knob in a RAG system, and it is where most bad RAG
results come from - ahead of the model, and usually ahead of the embedding
choice too.
"""

from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Chosen against the embedding model's 256-word-piece (~1,000 character) input
# limit: a chunk larger than that window is partly invisible to search. See the
# note at the top of retrieval.py.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass
class IngestReport:
    files: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    documents: int = 0
    chunks: int = 0
    total_in_store: int = 0


def _load_file(path: Path) -> list[Document]:
    """CONCEPT: document loaders

    A loader turns some source into `Document` objects - `page_content` plus a
    `metadata` dict - and that uniform output is the whole point: nothing
    downstream cares whether the text came from Markdown, a PDF, a website, or
    Slack. LangChain ships loaders for hundreds of sources (they used to live in
    `langchain_community`, which is now being sunset in favour of per-integration
    packages), but for plain text and PDFs a loader is thin enough to write
    directly, which keeps the dependency list short and makes the Document
    interface concrete.

    PDFs yield one Document per page with the page number in metadata, which is
    what lets a citation name a page.
    """
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": str(path), "page": number},
            )
            for number, page in enumerate(reader.pages)
        ]
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": str(path)},
        )
    ]


def load_documents(path: str | Path) -> tuple[list[Document], IngestReport]:
    """Load one file, or every supported file in a directory tree."""
    root = Path(path).expanduser().resolve()
    report = IngestReport()
    if not root.exists():
        raise FileNotFoundError(f"No such path: {root}")

    candidates = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    documents: list[Document] = []
    for candidate in candidates:
        if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
            report.skipped.append(str(candidate))
            continue
        loaded = _load_file(candidate)
        # Normalize `source` to the absolute path. Loaders set it differently,
        # and re-ingesting relies on it to find a file's existing chunks.
        for document in loaded:
            document.metadata["source"] = str(candidate)
        documents.extend(loaded)
        report.files.append(str(candidate))

    report.documents = len(documents)
    return documents, report


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """CONCEPT: RecursiveCharacterTextSplitter

    It tries a list of separators in order - paragraph break, line break, space,
    then bare character - and only falls back to a cruder one when a piece is
    still too big. The effect is that it cuts at the most natural boundary that
    fits, so chunks tend to hold whole paragraphs or sentences instead of ending
    mid-word.

    `chunk_overlap` repeats the tail of one chunk at the head of the next. A
    sentence sitting on a boundary would otherwise be split across two chunks and
    be fully present in neither, so neither would match a question about it.
    Overlap is insurance against exactly that, paid for in duplicated tokens.

    The metadata carries the source path through the split, plus a chunk index,
    which is what later lets an answer cite where it came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # records each chunk's offset in the original text
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index
    return chunks


def ingest_path(
    path: str | Path,
    *,
    store: Chroma | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IngestReport:
    """Load, chunk, embed, and store everything under `path`."""
    from research_copilot.retrieval import count_chunks, get_vector_store

    documents, report = load_documents(path)
    if not documents:
        store = store or get_vector_store()
        report.total_in_store = count_chunks(store)
        return report

    chunks = chunk_documents(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    store = store or get_vector_store()

    # Re-ingesting a file replaces its chunks instead of duplicating them.
    # Without this, running `ingest` twice doubles every chunk, and retrieval
    # then returns the same passage several times and crowds out everything else.
    for source in report.files:
        store.delete(where={"source": source})

    # add_documents embeds every chunk (locally, in batches) and writes vectors,
    # text, and metadata to the collection.
    store.add_documents(chunks)

    report.chunks = len(chunks)
    report.total_in_store = count_chunks(store)
    return report

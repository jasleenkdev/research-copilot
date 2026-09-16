"""Retrieval: embeddings, the vector store, and the RAG chain.

CONCEPT: what RAG actually is
Retrieval-Augmented Generation = search your documents for passages relevant to
the question, paste those passages into the prompt, and ask the model to answer
from them. The model is not fine-tuned or modified in any way. RAG is prompt
construction with a search step in front of it.

CONCEPT: embeddings
An embedding maps text to a vector positioned so that similar meanings land near
each other. Search becomes "embed the question, return the chunks whose vectors
are nearest". That finds passages about the same idea even when they share no
keywords with the question - and, less happily, sometimes returns passages that
merely sound alike. Retrieval quality is a property of the embedding model and
the chunking, not of Claude.

Why sentence-transformers/all-MiniLM-L6-v2, running locally:
  - Anthropic doesn't serve an embeddings API, so the embedder is never the same
    model that writes the answer. (Anthropic recommends Voyage AI, which needs
    another paid key.)
  - This model is free, needs no API key, and runs offline after a ~90 MB first
    download cached under ~/.cache/huggingface. Good for a learning project
    where you want to re-ingest repeatedly without paying per token.
  - The tradeoffs: it is small, so it is weaker than paid embeddings on
    technical or nuanced retrieval, and it truncates input at 256 word pieces
    (~1,000 characters). Anything past that is silently ignored when embedding.
    That limit is the real reason ingest.py chunks at 800 characters: a chunk
    bigger than the embedding window is partly invisible to search, even though
    the full text still gets pasted into the prompt.
  - RESEARCH_COPILOT_EMBEDDING_MODEL swaps it. Vectors from different models are
    not comparable, so changing the model means re-ingesting from scratch.
"""

from collections.abc import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from research_copilot.config import get_settings
from research_copilot.models import get_chat_model
from research_copilot.prompts import RAG_PROMPT

# Loading a sentence-transformers model takes a few seconds, so keep one per
# process rather than rebuilding it for every call.
_EMBEDDINGS_CACHE: dict[str, Embeddings] = {}


def get_embeddings(model_name: str | None = None) -> Embeddings:
    name = model_name or get_settings().embedding_model
    if name not in _EMBEDDINGS_CACHE:
        _EMBEDDINGS_CACHE[name] = HuggingFaceEmbeddings(model_name=name)
    return _EMBEDDINGS_CACHE[name]


def get_vector_store(
    *,
    embeddings: Embeddings | None = None,
    persist_directory: str | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Open (or create) the on-disk Chroma collection.

    CONCEPT: the vector store
    It stores each chunk's text, its vector, and its metadata, and answers
    nearest-neighbour queries. Chroma runs in-process and persists to a local
    directory - no server to run. Passing persist_directory=None gives an
    in-memory store, which is what the tests use.
    """
    settings = get_settings()
    return Chroma(
        collection_name=collection_name or settings.collection_name,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=(
            persist_directory
            if persist_directory is not None
            else str(settings.chroma_dir)
        ),
    )


def count_chunks(store: Chroma) -> int:
    return store._collection.count()


def get_retriever(
    *, store: Chroma | None = None, k: int | None = None
) -> VectorStoreRetriever:
    """CONCEPT: a retriever is just a Runnable of `str -> list[Document]`.

    That uniform interface is why it drops straight into an LCEL chain, and why
    swapping Chroma for FAISS, or for a keyword search, changes nothing
    downstream. `k` is how many chunks come back: too few and the answer may be
    missing its evidence, too many and the prompt fills with noise (and tokens).
    """
    store = store or get_vector_store()
    return store.as_retriever(search_kwargs={"k": k or get_settings().retrieval_k})


def describe_source(document: Document) -> str:
    source = document.metadata.get("source", "unknown")
    page = document.metadata.get("page")
    return f"{source}, page {page + 1}" if isinstance(page, int) else str(source)


def format_docs(documents: Sequence[Document]) -> str:
    """Render retrieved chunks as numbered excerpts the prompt can cite.

    The numbering is what makes the RAG prompt's "cite [1], [2]" rule usable, and
    it's what lets you check an answer against its evidence by hand.
    """
    if not documents:
        return "(no excerpts retrieved)"
    return "\n\n".join(
        f"[{i}] ({describe_source(doc)})\n{doc.page_content.strip()}"
        for i, doc in enumerate(documents, start=1)
    )


def build_rag_chain(
    *, model: BaseChatModel | None = None, retriever: VectorStoreRetriever | None = None
) -> Runnable[str, dict]:
    """question (str) -> {question, docs, context, answer}.

    CONCEPT: RunnableParallel and RunnablePassthrough.assign
    Phase 1's chains were a straight line. This one forks and rejoins:

        RunnableParallel(question=passthrough, docs=retriever)
            runs both branches on the same input - the question flows through
            untouched while the retriever turns it into Documents
        .assign(context=...)
            adds a key computed from the dict so far, keeping everything else
        .assign(answer=RAG_PROMPT | model | StrOutputParser())
            adds the answer, with the sub-chain reading {context} and {question}
            straight out of that dict

    The result keeps `docs` alongside `answer`, so the CLI can show which
    excerpts produced the answer. Keeping retrieval inside the chain (rather than
    retrieving separately and calling a chain) is what makes the whole thing one
    traced unit in LangSmith - you can see the retrieved chunks next to the
    answer they produced.
    """
    model = model or get_chat_model()
    retriever = retriever if retriever is not None else get_retriever()
    return (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        | RunnablePassthrough.assign(context=lambda payload: format_docs(payload["docs"]))
        | RunnablePassthrough.assign(answer=RAG_PROMPT | model | StrOutputParser())
    )

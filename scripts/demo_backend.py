"""A scripted backend for frontend work: the real API, graph and event stream,
with scripted model replies. No API key, no network, no tokens spent.

    python scripts/demo_backend.py serve            # API on http://127.0.0.1:8000
    python scripts/demo_backend.py record out.json  # re-record frontend/src/demo/recorded-run.json

Every question gets the same scripted run: research, write, the Critic rejects
an unsupported citation, rewrite, approve. Use it to work on the UI; use the
real server (uvicorn research_copilot.api:app) to work on answers.
"""
import json
import sys

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from research_copilot.agents.supervisor import SupervisorDecision
from research_copilot.api import create_app
from research_copilot.multi_agent_graph import build_multi_agent_graph


class NoStream(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class Streams(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ScriptedSupervisor(NoStream):
    decisions: list = []

    def with_structured_output(self, schema, **kwargs):
        it = iter(self.decisions)
        return RunnableLambda(lambda _msgs: next(it))


@tool("search_arxiv")
def search(query: str, max_results: int = 5) -> str:
    """Search arXiv."""
    return "results"


@tool("verify_citation")
def verify(arxiv_id: str) -> str:
    """Verify an arXiv id."""
    titles = {"2309.15217": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
              "2311.09476": "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems"}
    return f"FOUND: {arxiv_id} - {titles.get(arxiv_id, 'Unknown')}"


def tc(q, i):
    return AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": q}, "id": f"c{i}", "type": "tool_call"}])


NOTES = """- Findings:
  - RAGAS scores a RAG pipeline without human references, on faithfulness, answer relevance and context relevance (http://arxiv.org/abs/2309.15217).
  - ARES trains lightweight LM judges on synthetic data and uses prediction-powered inference to bound their error (http://arxiv.org/abs/2311.09476).
- Sources: arXiv 2309.15217, arXiv 2311.09476
- Gaps: no head-to-head comparison of the two on the same benchmark."""

DRAFT1 = """RAG pipelines are usually evaluated on two halves separately: **retrieval** (did the right context come back?) and **generation** (is the answer grounded in that context and does it address the question?).

- **RAGAS** [2309.15217] scores pipelines without human-written references, along three axes: *faithfulness* (claims supported by the retrieved context), *answer relevance*, and *context relevance*.
- **ARES** [2311.09476] trains small LM judges on synthetic queries and uses prediction-powered inference to put confidence intervals on their scores.
- **TruLens** [2401.00001] adds a "RAG triad" dashboard for production monitoring.

In practice teams combine an automated framework like these with a small human-labelled set to calibrate the judges."""

DRAFT2 = """RAG pipelines are usually evaluated on two halves separately: **retrieval** (did the right context come back?) and **generation** (is the answer grounded in that context and does it address the question?).

- **RAGAS** [2309.15217] scores pipelines without human-written references, along three axes: *faithfulness* (claims supported by the retrieved context), *answer relevance*, and *context relevance*.
- **ARES** [2311.09476] trains small LM judges on synthetic queries and uses prediction-powered inference to put confidence intervals on their scores, so a handful of human labels can correct the judges' bias.

The two are complementary rather than competing: RAGAS is quick to run with no training, while ARES costs more setup and gives error bounds. No published head-to-head comparison on one benchmark was found in this search."""

D = SupervisorDecision
decisions = [
    D(rationale="No research notes yet; the question needs evidence on evaluation frameworks before anything can be written.", next="researcher",
      researcher_brief=""),
    D(rationale="Notes cover two frameworks (RAGAS, ARES) with sources, which is enough to answer the question.", next="writer"),
    D(rationale="A draft exists and the Critic is on the roster; it should be reviewed before finishing.", next="critic"),
    D(rationale="The Critic flags TruLens [2401.00001], which is not in the research notes. That is a writing problem: the Writer added a source, so the Writer fixes it.", next="writer"),
    D(rationale="Revised draft drops the unsupported citation; it needs a fresh review.", next="critic"),
    D(rationale="The Critic approved the revised draft and every citation was found in the notes.", next="finish"),
]


def factory(policy, saver):
    return build_multi_agent_graph(
        **{**policy, "memory_strategy": "none"},
        supervisor_model=ScriptedSupervisor(responses=[AIMessage(content="x")], decisions=decisions),
        researcher_model=NoStream(responses=[tc("RAG evaluation framework", 1), tc("automated evaluation retrieval augmented generation judges", 2), AIMessage(content=NOTES)]),
        writer_model=Streams(messages=iter([AIMessage(content=DRAFT1), AIMessage(content=DRAFT1), AIMessage(content=DRAFT2), AIMessage(content=DRAFT2)])),
        critic_model=NoStream(responses=[
            AIMessage(content="REJECT\nWriting: cites TruLens (2401.00001), which is not in the research notes - the Writer added it. Remove it or send the Researcher to support it."),
            AIMessage(content="APPROVE\nEvery claim maps to the notes; both citations verified."),
        ]),
        tools=[search], critic_tools=[verify], checkpointer=saver,
    )


def record(path: str) -> None:
    from fastapi.testclient import TestClient

    app = create_app(saver=MemorySaver(), graph_factory=factory, model_check=None)
    with TestClient(app) as c:
        r = c.post("/api/runs", json={"question": "How are RAG pipelines evaluated?", "critic": True, "max_revisions": 2})
        events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    with open(path, "w") as f:
        json.dump(events, f, indent=1)
    print(f"{len(events)} events -> {path} (ends: {events[-1]['type']})")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "record":
        record(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "serve":
        import uvicorn

        uvicorn.run(create_app(saver=MemorySaver(), graph_factory=factory, model_check=None), port=8000)
    else:
        print(__doc__)
        sys.exit(2)

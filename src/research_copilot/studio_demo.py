"""An offline demo of the multi-agent graph, for LangGraph Studio. (6.4)

NOT the real system. It is the real multi-agent graph - the same
`build_multi_agent_graph`, the same nodes, subgraphs, guards and routing
functions - with every model replaced by a script and every tool by a stub.
It exists so the graph can be *run* in Studio without an ANTHROPIC_API_KEY or
network access, and so that every run takes the same instructive path:

    researcher (one search, then notes)
    -> writer (draft 1, which cites a source the notes don't contain)
    -> supervisor proposes finish -> OVERRIDDEN to critic
    -> critic verifies, REJECTS ("writing problem")
    -> start_revision -> supervisor classifies -> writer (draft 2)
    -> critic verifies, APPROVES -> finalize_answer

That path exercises both subgraphs (and their private channels), a guard
override, a rejection routed back through the Supervisor, a revision reset,
and a per-agent budget - everything Phase 6 added - in one run.

Registered in langgraph.json as `multi_agent_demo`, next to `multi_agent`
(the real graph, which Studio can draw without a key but cannot run).

Every script below is consumed exactly once per run, so a fresh graph or a
cycling fake stays aligned run after run.
"""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import tool

from research_copilot.agents.supervisor import SupervisorDecision
from research_copilot.multi_agent_graph import build_multi_agent_graph


class _ScriptedToolModel(FakeMessagesListChatModel):
    """A scripted chat model that accepts bind_tools (and ignores it)."""

    def bind_tools(self, tools, **kwargs):
        return self


def _call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class _ScriptedSupervisor:
    """Stands in for a model's with_structured_output: one decision per call, cycling."""

    def __init__(self, decisions: list[SupervisorDecision]):
        self._decisions = decisions
        self._i = 0

    def with_structured_output(self, schema, **kwargs) -> Runnable:
        def decide(_messages):
            decision = self._decisions[self._i % len(self._decisions)]
            self._i += 1
            return decision

        return RunnableLambda(decide, name="scripted_supervisor")


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Stub arXiv search (offline demo)."""
    return (
        "[1] RAGAS: Automated Evaluation of Retrieval Augmented Generation\n"
        "    URL: http://arxiv.org/abs/2309.15217\n"
        "    Abstract: reference-free metrics for faithfulness and relevance."
    )


@tool
def verify_citation(arxiv_id: str) -> str:
    """Stub citation check (offline demo)."""
    return f"FOUND: {arxiv_id} - (demo) a paper with this id exists"


def make_graph(config: dict | None = None) -> Runnable:
    researcher = _ScriptedToolModel(
        responses=[
            _call("search_arxiv", {"query": "RAG evaluation"}, "s1"),
            AIMessage(
                content=(
                    "- Findings: RAGAS scores faithfulness and answer relevance without "
                    "reference answers (RAGAS, http://arxiv.org/abs/2309.15217).\n"
                    "- Sources: RAGAS: Automated Evaluation of Retrieval Augmented "
                    "Generation, http://arxiv.org/abs/2309.15217\n"
                    "- Gaps: no head-to-head benchmark numbers found."
                )
            ),
        ]
    )
    writer = _ScriptedToolModel(
        responses=[
            AIMessage(
                content=(
                    "RAG pipelines are usually evaluated with RAGAS [2309.15217] and "
                    "ARES [2311.09476], which score faithfulness and relevance."
                )
            ),
            AIMessage(
                content=(
                    "RAG pipelines can be evaluated with RAGAS [2309.15217], which "
                    "scores faithfulness and answer relevance without reference "
                    "answers. No head-to-head benchmark numbers were found."
                )
            ),
        ]
    )
    critic = _ScriptedToolModel(
        responses=[
            _call("verify_citation", {"arxiv_id": "2309.15217"}, "v1"),
            _call("verify_citation", {"arxiv_id": "2311.09476"}, "v2"),
            AIMessage(
                content=(
                    "REJECT\nWriting problem: the draft cites ARES (2311.09476). The "
                    "paper exists, but it is not in the research notes - the Writer "
                    "added it. Remove it or have it researched."
                )
            ),
            _call("verify_citation", {"arxiv_id": "2309.15217"}, "v3"),
            AIMessage(content="APPROVE"),
        ]
    )
    supervisor = _ScriptedSupervisor(
        [
            SupervisorDecision(rationale="Nothing has been gathered yet.", next="researcher",
                               researcher_brief="how RAG pipelines are evaluated"),
            SupervisorDecision(rationale="The notes cover the question well enough to draft.", next="writer"),
            SupervisorDecision(rationale="The draft looks complete.", next="finish"),
            SupervisorDecision(
                rationale=(
                    "The critic says the Writer cited a paper that is not in the notes: a "
                    "writing problem, not missing evidence."
                ),
                next="writer",
            ),
            SupervisorDecision(rationale="Rewritten without the unsupported source; review it.", next="critic"),
        ]
    )
    return build_multi_agent_graph(
        researcher_model=researcher,
        writer_model=writer,
        critic_model=critic,
        supervisor_model=supervisor,
        tools=[search_arxiv],
        critic_tools=[verify_citation],
        enable_critic=True,
        max_revisions=2,
        memory_strategy="none",
    )

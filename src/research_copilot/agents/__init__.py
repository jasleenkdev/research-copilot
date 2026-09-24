"""Phase 6's agents, one module each.

An agent here is a prompt, the tools it may use, and the loop it runs, kept
together in one file so each can be read on its own. How the agents are wired
together is in multi_agent_graph.py. Which state each one may write is in
multi_agent_state.py.
"""

from research_copilot.agents.researcher import build_researcher
from research_copilot.agents.writer import make_writer

__all__ = ["build_researcher", "make_writer"]

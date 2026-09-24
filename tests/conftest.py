import os

# Tests use fake models and must never call Anthropic or LangSmith. This runs
# before research_copilot.config loads .env, and load_dotenv doesn't override
# variables that are already set.
os.environ["LANGSMITH_TRACING"] = "false"


import pytest


@pytest.fixture(autouse=True)
def _isolated_checkpoint_db(monkeypatch, tmp_path):
    """No test may write to the real data/checkpoints.sqlite3.

    Added in 6.4, when `multi-agent` gained a checkpointer that defaults to
    sqlite (Phase 4's default): CLI tests written for 6.1-6.3, which never
    passed --checkpointer, silently started writing threads into the user's
    database. Individual tests may still point it elsewhere.
    """
    monkeypatch.setenv("RESEARCH_COPILOT_CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite3"))

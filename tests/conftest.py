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


@pytest.fixture(autouse=True)
def _no_real_provider_access(monkeypatch):
    """No test may reach a real model provider or use a real key. (Part D)

    Found when Part D added a startup preflight (`check_model_available`): the
    CLI tests passed in the full suite only because an earlier test had left
    RESEARCH_COPILOT_PROVIDER=groq in the environment, and the real
    GROQ_API_KEY from .env was loaded - so the preflight made real network
    calls from inside the test suite. Every test now starts with no real keys,
    the default provider, and the preflight stubbed. Tests of the preflight
    itself patch it back explicitly.
    """
    for key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "RESEARCH_COPILOT_PROVIDER",
                "RESEARCH_COPILOT_FALLBACK_MODEL", "RESEARCH_COPILOT_MAX_REQUEST_TOKENS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("research_copilot.cli.check_model_available", lambda *a, **k: None)

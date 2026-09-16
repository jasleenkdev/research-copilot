"""Configuration: load .env once and expose typed settings.

Why a config module at all? LangChain, LangSmith, and the Anthropic SDK read
their settings straight from environment variables (ANTHROPIC_API_KEY,
LANGSMITH_TRACING, LANGSMITH_API_KEY, ...). If .env is loaded in one place before
any model is built, auth and tracing work everywhere else with no keys passed
around. Enabling LangSmith tracing needs no code at all; the env vars are enough.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# override=False: a variable already exported in your shell wins over .env.
# (The tests rely on this to force tracing off.)
load_dotenv(override=False)

DEFAULT_MODEL = "claude-opus-5"


@dataclass(frozen=True)
class Settings:
    model: str
    tracing_enabled: bool
    langsmith_api_key_set: bool
    langsmith_project: str | None


def get_settings() -> Settings:
    return Settings(
        model=os.getenv("RESEARCH_COPILOT_MODEL") or DEFAULT_MODEL,
        tracing_enabled=os.getenv("LANGSMITH_TRACING", "").lower() == "true",
        langsmith_api_key_set=bool(os.getenv("LANGSMITH_API_KEY")),
        langsmith_project=os.getenv("LANGSMITH_PROJECT"),
    )


def require_anthropic_key() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example)."
        )

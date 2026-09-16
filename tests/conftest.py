import os

# Tests use fake models and must never call Anthropic or LangSmith. This runs
# before research_copilot.config loads .env, and load_dotenv doesn't override
# variables that are already set.
os.environ["LANGSMITH_TRACING"] = "false"

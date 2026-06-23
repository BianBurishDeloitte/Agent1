import os
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "claude-haiku-4-5-20251001")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))

# ── LangSmith ─────────────────────────────────────────────────────────────────
LANGSMITH_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "Agent1")
LANGSMITH_TRACING: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

if LANGSMITH_TRACING and LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# ── Validation ────────────────────────────────────────────────────────────────
if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set. Copy .env.example → .env and add your key "
        "from https://console.anthropic.com/settings/keys"
    )

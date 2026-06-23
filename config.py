import os
from dotenv import load_dotenv

load_dotenv()

# ── Google Gemini ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))

# ── LangSmith ─────────────────────────────────────────────────────────────────
LANGSMITH_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "Agent-Team-Meeting")
LANGSMITH_TRACING: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

if LANGSMITH_TRACING and LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# ── Validation ────────────────────────────────────────────────────────────────
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com "
        "then add it to your .env file."
    )

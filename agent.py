from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY, MODEL_NAME, TEMPERATURE, LANGSMITH_PROJECT, LANGSMITH_TRACING
from tools import get_tools

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Always use the available tools when they are relevant to the question. "
    "Think carefully before answering."
)


def create_agent():
    """Return a LangGraph ReAct agent backed by Gemini."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
    )
    return create_react_agent(
        model=llm,
        tools=get_tools(),
        prompt=_SYSTEM_PROMPT,
    )

from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from config import (
    GOOGLE_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
    MODEL_NAME,
    TEMPERATURE,
)
from tools import get_tools

# Standard ReAct prompt — requires {tools}, {tool_names}, {input}, {agent_scratchpad}
_REACT_PROMPT = """Answer the following questions as best you can. \
You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the tool to use, should be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def create_agent() -> AgentExecutor:
    """Build and return a configured ReAct AgentExecutor backed by Gemini."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
    )

    tools = get_tools()
    prompt = PromptTemplate.from_template(_REACT_PROMPT)
    react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10,
        tags=["react-agent", "gemini", "demo"],
    )

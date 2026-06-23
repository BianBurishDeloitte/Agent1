import ast
import operator
import random

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Safe arithmetic evaluator (no exec/eval risks)
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    def _walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_walk(node.operand))
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    tree = ast.parse(expr.strip(), mode="eval")
    return _walk(tree.body)


# ---------------------------------------------------------------------------
# Tool 1 — Calculator
# ---------------------------------------------------------------------------


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Input must be a valid arithmetic expression such as '2 + 2', '15 * 4 / 3',
    or '2 ** 8'. Do not include units or text — numbers and operators only."""
    try:
        raw = _safe_eval(expression)
        result = int(raw) if isinstance(raw, float) and raw.is_integer() else raw
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# Tool 2 — Weather (mock)
# ---------------------------------------------------------------------------

_WEATHER_DB: dict[str, dict] = {
    "london":   {"condition": "Overcast",       "temp_c": 14, "humidity": 82, "wind": 18},
    "new york": {"condition": "Partly Cloudy",  "temp_c": 22, "humidity": 60, "wind": 12},
    "tel aviv": {"condition": "Sunny",          "temp_c": 31, "humidity": 65, "wind": 20},
    "paris":    {"condition": "Light Rain",     "temp_c": 16, "humidity": 88, "wind": 25},
    "tokyo":    {"condition": "Clear",          "temp_c": 27, "humidity": 70, "wind":  8},
    "sydney":   {"condition": "Sunny",          "temp_c": 19, "humidity": 55, "wind": 15},
    "berlin":   {"condition": "Cloudy",         "temp_c": 12, "humidity": 76, "wind": 22},
    "dubai":    {"condition": "Hot and Sunny",  "temp_c": 41, "humidity": 45, "wind": 10},
    "moscow":   {"condition": "Snowy",          "temp_c": -3, "humidity": 90, "wind": 30},
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.
    Input should be a city name such as 'London', 'New York', or 'Tel Aviv'.
    Returns temperature, humidity, wind speed, and weather condition."""
    data = _WEATHER_DB.get(city.lower().strip())

    if data is None:
        temp_c = random.randint(10, 35)
        data = {
            "condition": "Partly Cloudy",
            "temp_c": temp_c,
            "humidity": random.randint(40, 80),
            "wind": random.randint(5, 30),
        }
        note = "\n  (Simulated — city not in local database)"
    else:
        note = ""

    temp_f = round(data["temp_c"] * 9 / 5 + 32)
    return (
        f"Weather in {city.title()}:\n"
        f"  Condition   : {data['condition']}\n"
        f"  Temperature : {data['temp_c']}°C  /  {temp_f}°F\n"
        f"  Humidity    : {data['humidity']}%\n"
        f"  Wind        : {data['wind']} km/h"
        f"{note}"
    )


# ---------------------------------------------------------------------------
# Tool 3 — Search (mock)
# ---------------------------------------------------------------------------

_SEARCH_DB: dict[str, str] = {
    "python": (
        "Python is a high-level, interpreted programming language created by Guido van Rossum "
        "in 1991. It is known for its clean syntax and is widely used in web development, "
        "data science, AI/ML, and automation."
    ),
    "langchain": (
        "LangChain is an open-source framework for building LLM-powered applications. "
        "It provides abstractions for chaining model calls, managing memory, and integrating "
        "external tools and data sources. Latest stable version: 0.3.x."
    ),
    "openai": (
        "OpenAI is an AI research company that developed GPT-4, ChatGPT, DALL-E, and Whisper. "
        "Founded in 2015, it focuses on building safe and beneficial artificial general intelligence."
    ),
    "machine learning": (
        "Machine learning (ML) is a subset of AI where systems learn patterns from data without "
        "explicit programming. Core paradigms include supervised learning, unsupervised learning, "
        "and reinforcement learning."
    ),
    "israel": (
        "Israel is a Middle Eastern country with ~9.5 million people. Known as the 'Start-up Nation', "
        "it has one of the highest densities of tech companies per capita globally. "
        "Capital: Jerusalem. Currency: New Israeli Shekel (NIS)."
    ),
    "deloitte": (
        "Deloitte is one of the Big Four professional services firms, offering audit, consulting, "
        "tax, and advisory services to clients in 150+ countries with over 415,000 professionals."
    ),
    "artificial intelligence": (
        "Artificial Intelligence (AI) is the field of computer science focused on creating systems "
        "that simulate human intelligence. It includes machine learning, NLP, computer vision, "
        "robotics, and expert systems."
    ),
    "climate change": (
        "Climate change refers to long-term shifts in global temperatures driven primarily by human "
        "activities such as burning fossil fuels and deforestation, leading to rising sea levels, "
        "extreme weather events, and ecosystem disruptions."
    ),
    "react": (
        "ReAct (Reasoning + Acting) is a prompting strategy for LLM agents. The model alternates "
        "between producing reasoning traces (Thoughts) and taking actions (tool calls), allowing "
        "it to ground decisions in real observations."
    ),
}


@tool
def search(query: str) -> str:
    """Search for information about a topic or answer a general knowledge question.
    Input should be a concise topic or query string such as 'Python', 'climate change',
    or 'LangChain'. Returns a brief informational summary."""
    query_lower = query.lower().strip()
    for key, text in _SEARCH_DB.items():
        if key in query_lower or query_lower in key:
            return f"Search result for '{query}':\n{text}"

    return (
        f"No specific result found for '{query}' in the local knowledge base. "
        "In production this would query a live search API or vector store."
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def get_tools() -> list:
    return [calculator, get_weather, search]

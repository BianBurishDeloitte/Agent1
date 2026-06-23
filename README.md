# LangChain ReAct Agent Demo

A production-quality demo agent built with **LangChain 0.3**, **OpenAI**, and **Rich** for structured, colour-coded terminal output — or optionally a **Streamlit** web UI.

---

## Project structure

```
Agent Team Meeting/
├── config.py          # env loading (dotenv)
├── tools.py           # Calculator, Weather, Search tools
├── logger.py          # Rich-formatted callback handler
├── agent.py           # AgentExecutor factory
├── main.py            # Interactive CLI entry point
├── streamlit_app.py   # Streamlit web UI (bonus)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick start

### 1 · Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

### 3 · Configure your API key

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

Open `.env` and replace `sk-...` with your real OpenAI key.

### 4a · Run the CLI

```bash
python main.py
```

### 4b · Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## VS Code setup

1. Open the folder in VS Code: **File → Open Folder**
2. Select the `.venv` interpreter: `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `.venv`
3. Open the integrated terminal: `` Ctrl+` ``
4. Run `python main.py` (or the streamlit command above)

---

## Example prompts for live demo

| Category    | Prompt |
|-------------|--------|
| Math        | `What is 1337 multiplied by 42?` |
| Math        | `What is 2 to the power of 10, minus 24?` |
| Math        | `What is 25 percent of 840?` |
| Weather     | `What's the weather like in Tel Aviv?` |
| Weather     | `Compare the weather in London and Dubai` |
| Search      | `Tell me about LangChain` |
| Search      | `What is artificial intelligence?` |
| Multi-tool  | `What is the weather in Tokyo, and what is 37 degrees Celsius in Fahrenheit?` |

---

## Architecture notes

- **ReAct loop** — the agent alternates Thought → Action → Observation until it can produce a Final Answer.
- **No deprecated APIs** — uses `create_react_agent` + `AgentExecutor` (not the legacy `initialize_agent`).
- **Callbacks via `invoke` config** — keeps the agent stateless; both CLI and Streamlit pass their own logger per call.
- **Safe calculator** — uses Python's `ast` module; no `eval()` with arbitrary code.

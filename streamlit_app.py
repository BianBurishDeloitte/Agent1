import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler

from agent import create_agent
from config import LANGSMITH_PROJECT, LANGSMITH_TRACING, MODEL_NAME

# ---------------------------------------------------------------------------
# Streamlit-specific callback handler
# ---------------------------------------------------------------------------


class StreamlitStepLogger(BaseCallbackHandler):
    """Write each ReAct step into a Streamlit container in real time."""

    def __init__(self, container) -> None:
        super().__init__()
        self._container = container
        self._step = 0

    def _thought(self, log: str) -> str:
        log = log.strip()
        raw = log.split("\nAction:")[0] if "Action:" in log else log
        return raw.replace("Thought:", "").strip()

    def on_agent_action(self, action, **kwargs) -> None:
        self._step += 1
        thought = self._thought(action.log)
        with self._container:
            st.markdown(f"#### Step {self._step}")
            if thought:
                st.info(f"💭 **Thought:** {thought}")
            st.warning(
                f"⚙️ **Action:** `{action.tool}`\n\n"
                f"**Input:** `{action.tool_input}`"
            )

    def on_tool_end(self, output, **kwargs) -> None:
        with self._container:
            st.success(f"👁 **Observation:**\n\n{output}")

    def on_tool_error(self, error, **kwargs) -> None:
        with self._container:
            st.error(f"❌ **Tool error:** {error}")


# ---------------------------------------------------------------------------
# Agent (cached — created once per Streamlit session)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading agent…")
def get_agent():
    return create_agent()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="LangChain ReAct Agent",
        page_icon="🤖",
        layout="wide",
    )

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🤖 ReAct Agent")
        st.caption("Powered by LangChain + OpenAI")

        st.divider()
        st.markdown("**Model**")
        st.code(MODEL_NAME, language=None)

        st.divider()
        st.markdown("**LangSmith tracing**")
        if LANGSMITH_TRACING:
            st.success(f"● ON — project: `{LANGSMITH_PROJECT}`")
            st.markdown("[Open LangSmith ↗](https://smith.langchain.com)")
        else:
            st.warning("○ OFF — add keys to .env to enable")

        st.divider()
        st.markdown("**Available tools**")
        st.markdown("- 🔢 **Calculator** — arithmetic expressions")
        st.markdown("- 🌤 **Weather** — city weather lookup")
        st.markdown("- 🔍 **Search** — general knowledge")

        st.divider()
        st.markdown("**Try these prompts**")
        for ex in [
            "What is 1337 × 42?",
            "Weather in Tokyo?",
            "Tell me about LangChain",
            "What is 2 ** 10 - 24?",
            "Weather in Dubai?",
            "Search for artificial intelligence",
            "Weather in Tokyo, and 37°C in °F?",
        ]:
            st.code(ex, language=None)

        st.divider()
        if st.button("🗑 Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Main chat area ─────────────────────────────────────────────────────
    st.title("LangChain ReAct Agent Demo")
    st.caption(
        "Live agent reasoning: Thought → Action → Observation → Final Answer"
        + (f"  •  LangSmith project: **{LANGSMITH_PROJECT}**" if LANGSMITH_TRACING else "")
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask the agent anything…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            steps_expander = st.expander("🧠 Agent reasoning", expanded=True)
            logger = StreamlitStepLogger(steps_expander)
            agent = get_agent()

            with st.spinner("Thinking…"):
                try:
                    result = agent.invoke(
                        {"input": prompt},
                        config={"callbacks": [logger]},
                    )
                    answer = result.get("output", "No response generated.")
                except Exception as exc:
                    answer = f"⚠️ Error: {exc}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

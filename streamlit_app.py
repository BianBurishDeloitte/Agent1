import json

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from agent import create_agent
from config import LANGSMITH_PROJECT, LANGSMITH_TRACING, MODEL_NAME


@st.cache_resource(show_spinner="Loading agent…")
def get_agent():
    return create_agent()


def _stream_agent(agent, prompt: str, steps_container) -> str:
    """Stream agent steps into the Streamlit container; return the final answer."""
    step = 0
    final_answer = ""

    for chunk in agent.stream(
        {"messages": [("human", prompt)]},
        stream_mode="updates",
    ):
        for node_name, node_output in chunk.items():
            for msg in node_output.get("messages", []):

                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        step += 1
                        args = (json.dumps(tc["args"], ensure_ascii=False)
                                if isinstance(tc["args"], dict) else str(tc["args"]))
                        with steps_container:
                            st.markdown(f"**Step {step}**")
                            st.warning(f"⚙️ **Action:** `{tc['name']}`\n\n**Input:** `{args}`")

                elif isinstance(msg, ToolMessage):
                    with steps_container:
                        st.success(f"👁 **Observation:**\n\n{msg.content}")

                elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_answer = msg.content

    return final_answer or "No response generated."


def main() -> None:
    st.set_page_config(page_title="Agent Team Meeting", page_icon="🤖", layout="wide")

    with st.sidebar:
        st.title("🤖 Agent Team Meeting")
        st.caption("LangChain + Gemini 2.5 Flash")
        st.divider()
        st.markdown("**Model**")
        st.code(MODEL_NAME, language=None)
        st.divider()
        st.markdown("**LangSmith**")
        if LANGSMITH_TRACING:
            st.success(f"● ON — `{LANGSMITH_PROJECT}`")
            st.markdown("[Open LangSmith ↗](https://smith.langchain.com)")
        else:
            st.warning("○ OFF")
        st.divider()
        st.markdown("**Tools**")
        st.markdown("- 🔢 Calculator\n- 🌤 Weather\n- 🔍 Search")
        st.divider()
        st.markdown("**Try these**")
        for ex in [
            "What is 1337 × 42?",
            "Weather in Tokyo?",
            "Tell me about LangChain",
            "What is 25% of 840?",
            "Weather in Dubai and 37°C in °F?",
        ]:
            st.code(ex, language=None)
        st.divider()
        if st.button("🗑 Clear history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("Agent Team Meeting — ReAct Agent Demo")
    st.caption("Powered by LangChain + Google Gemini 2.5 Flash + LangSmith")

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
            agent = get_agent()
            with st.spinner("Thinking…"):
                try:
                    answer = _stream_agent(agent, prompt, steps_expander)
                except Exception as exc:
                    answer = f"⚠️ Error: {exc}"
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

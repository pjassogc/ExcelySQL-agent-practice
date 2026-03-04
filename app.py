"""
app.py — Streamlit web UI for the Excel Analyst Agent.

Usage
-----
    streamlit run app.py

The app lets users upload an Excel file via the sidebar, then chat with the
LangGraph ReAct agent in a persistent, multi-turn session. All agent state
is stored in st.session_state so it survives Streamlit reruns.

Teaching points:
    - Streamlit re-executes the entire script on every interaction; session_state
      is the escape hatch for anything that must persist between reruns.
    - We reuse agent.py, tools.py, state.py unchanged — Streamlit is just a UI layer.
    - thread_id is a fixed UUID per session, giving automatic multi-turn memory.
    - Tool calls are surfaced in an expander so users can see the agent's reasoning.
"""

import os
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

from agent import build_agent
from tools import build_tools, _pick_engine

# Load .env before any LangChain/OpenAI imports execute
load_dotenv()

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Excel Analyst Agent",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
# These keys are set once; subsequent reruns skip the `not in` branches.

if "graph" not in st.session_state:
    st.session_state.graph = None          # CompiledGraph | None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []         # list[dict] — {role, content, tool_calls}
if "excel_path" not in st.session_state:
    st.session_state.excel_path = None     # str path to the temp file
if "df" not in st.session_state:
    st.session_state.df = None             # pd.DataFrame
if "all_sheets" not in st.session_state:
    st.session_state.all_sheets = {}       # dict[str, pd.DataFrame]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_build(uploaded_file) -> None:
    """
    Save the uploaded file to a temp location, load it into a DataFrame,
    build tools and agent, and store everything in session_state.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    # Write to a named temp file so pandas/openpyxl can seek on it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    engine = _pick_engine(tmp_path)
    df = pd.read_excel(tmp_path, engine=engine)
    all_sheets = pd.read_excel(tmp_path, sheet_name=None, engine=engine)

    tools = build_tools(df, tmp_path)
    graph = build_agent(tools)

    st.session_state.graph = graph
    st.session_state.excel_path = tmp_path
    st.session_state.df = df
    st.session_state.all_sheets = all_sheets
    # Reset conversation when a new file is loaded
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())


def _reset_session() -> None:
    """Clear all session state (used by the 'New session' button)."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def extract_final_answer(result: dict) -> str:
    """
    Extract the last AI text message from the agent result dict.
    Mirrors the same function in chat.py.
    """
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return msg.content
    return "(No response generated)"


def _extract_tool_calls(result: dict) -> list[dict]:
    """
    Return a list of {tool_name, input, output} dicts from the result messages.
    Used to populate the Tool calls expander.
    """
    calls = []
    messages = result.get("messages", [])
    # Build a lookup from tool_call_id → ToolMessage output
    tool_outputs: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_outputs[msg.tool_call_id] = msg.content

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                calls.append({
                    "name": tc["name"],
                    "input": tc.get("args", {}),
                    "output": tool_outputs.get(tc["id"], ""),
                })
    return calls


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 Excel Analyst")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload an Excel file",
        type=["xlsx", "xls"],
        help="Supported formats: .xlsx, .xls",
    )

    if uploaded is not None:
        # Only rebuild when a *new* file is uploaded (compare by name+size)
        file_key = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_file_key") != file_key:
            with st.spinner("Loading file and building agent…"):
                _load_and_build(uploaded)
            st.session_state["_file_key"] = file_key
            st.success("Agent ready!")

    st.markdown("---")

    if st.button("🔄 New session", use_container_width=True):
        _reset_session()

    # Static info shown once a file is loaded
    if st.session_state.df is not None:
        df = st.session_state.df
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        st.markdown(f"**Model:** `{model_name}`")
        st.markdown(f"**Rows:** {len(df):,}  |  **Cols:** {len(df.columns)}")
        sheet_names = list(st.session_state.all_sheets.keys())
        st.markdown(f"**Sheets ({len(sheet_names)}):**")
        for name in sheet_names:
            n_rows = len(st.session_state.all_sheets[name])
            st.markdown(f"&nbsp;&nbsp;• `{name}` — {n_rows:,} rows")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if st.session_state.graph is None:
    # Welcome screen shown before any file is uploaded
    st.markdown(
        """
        # Welcome to Excel Analyst Agent 📊

        Upload an Excel file in the sidebar to get started.

        **What you can ask:**
        - *"What columns does this file have?"*
        - *"Show me the top 5 products by revenue"*
        - *"What's the average order value per region?"*
        - *"Plot a bar chart of monthly sales"*

        The agent uses a LangGraph ReAct loop with three tools:
        - **get_dataframe_info** — schema, stats, sample rows
        - **list_sheets** — workbook sheet names
        - **python_repl** — arbitrary pandas code via AST parsing

        Multi-turn memory is maintained within the session.
        """,
        unsafe_allow_html=False,
    )
    st.stop()

# File header
excel_name = Path(st.session_state.excel_path).name if st.session_state.excel_path else "unknown"
df = st.session_state.df
st.markdown(f"### 📁 `{excel_name}` &nbsp; — &nbsp; {len(df):,} rows × {len(df.columns)} cols")
st.markdown("---")

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("tool_calls"):
            with st.expander("🔧 Tool calls", expanded=False):
                for tc in entry["tool_calls"]:
                    st.markdown(f"**`{tc['name']}`**")
                    if tc["input"]:
                        st.json(tc["input"])
                    if tc["output"]:
                        st.code(tc["output"], language="text")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about your data…"):
    # Show the user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the agent
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking…"):
            try:
                result = st.session_state.graph.invoke(
                    {"messages": [("human", prompt)]},
                    config=config,
                )
                answer = extract_final_answer(result)
                tool_calls = _extract_tool_calls(result)
            except Exception as exc:
                answer = f"**Agent error:** {exc}"
                tool_calls = []

        st.markdown(answer)
        if tool_calls:
            with st.expander("🔧 Tool calls", expanded=False):
                for tc in tool_calls:
                    st.markdown(f"**`{tc['name']}`**")
                    if tc["input"]:
                        st.json(tc["input"])
                    if tc["output"]:
                        st.code(tc["output"], language="text")

    # Persist to message history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_calls": tool_calls,
    })

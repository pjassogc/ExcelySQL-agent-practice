# app.py
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

from agent import build_agent
from sql_agent import build_sql_agent
from sql_tools import build_sql_tools
from tools import build_tools, _pick_engine

load_dotenv()

st.set_page_config(
    page_title="Agente Analista",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Inicialización de session_state
# ---------------------------------------------------------------------------
if "agent_type" not in st.session_state:
    st.session_state.agent_type = "excel"
if "graph" not in st.session_state:
    st.session_state.graph = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
# Excel-specific
if "excel_path" not in st.session_state:
    st.session_state.excel_path = None
if "df" not in st.session_state:
    st.session_state.df = None
if "all_sheets" not in st.session_state:
    st.session_state.all_sheets = {}
# SQL-specific
if "db_path" not in st.session_state:
    st.session_state.db_path = None

CHART_PREFIX = "CHART:"


# ---------------------------------------------------------------------------
# Helpers — inicialización de agentes
# ---------------------------------------------------------------------------

def _load_excel_agent(uploaded_file) -> None:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    engine = _pick_engine(tmp_path)
    df = pd.read_excel(tmp_path, engine=engine)
    all_sheets = pd.read_excel(tmp_path, sheet_name=None, engine=engine)
    tools = build_tools(df, tmp_path)
    graph = build_agent(tools)

    st.session_state.update({
        "graph": graph,
        "excel_path": tmp_path,
        "df": df,
        "all_sheets": all_sheets,
        "messages": [],
        "thread_id": str(uuid.uuid4()),
    })


def _load_sql_agent(db_path: str) -> None:
    tools = build_sql_tools(db_path)
    graph = build_sql_agent(tools)
    st.session_state.update({
        "graph": graph,
        "db_path": db_path,
        "messages": [],
        "thread_id": str(uuid.uuid4()),
    })


def _reset_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ---------------------------------------------------------------------------
# Helpers — extracción de respuesta y tool calls
# ---------------------------------------------------------------------------

def extract_final_answer(result: dict) -> str:
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            if not getattr(msg, "tool_calls", None):
                return msg.content
    return "(Sin respuesta generada)"


def _extract_tool_calls(result: dict) -> list[dict]:
    calls = []
    tool_outputs: dict[str, str] = {}
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            tool_outputs[msg.tool_call_id] = msg.content
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                calls.append({
                    "name": tc["name"],
                    "input": tc.get("args", {}),
                    "output": tool_outputs.get(tc["id"], ""),
                })
    return calls


def _extract_chart_paths(tool_calls: list[dict]) -> list[str]:
    return [
        tc["output"][len(CHART_PREFIX):]
        for tc in tool_calls
        if isinstance(tc.get("output"), str) and tc["output"].startswith(CHART_PREFIX)
    ]


# ---------------------------------------------------------------------------
# Sidebar — secciones por tipo de agente
# ---------------------------------------------------------------------------

def _render_excel_sidebar() -> None:
    uploaded = st.file_uploader("Sube un archivo Excel", type=["xlsx", "xls"])
    if uploaded is not None:
        file_key = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_file_key") != file_key:
            with st.spinner("Cargando Excel y construyendo agente…"):
                _load_excel_agent(uploaded)
            st.session_state["_file_key"] = file_key
            st.success("¡Agente listo!")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"**Filas:** {len(df):,}  |  **Cols:** {len(df.columns)}")
        for name, sdf in st.session_state.all_sheets.items():
            st.markdown(f"&nbsp;&nbsp;• `{name}` — {len(sdf):,} filas")


def _render_sql_sidebar() -> None:
    db_path = Path("data/sample.db")
    if not db_path.exists():
        st.warning("No se encontró `data/sample.db`")
        st.code("python generate_db.py", language="bash")
        if st.button("⚡ Generar ahora", use_container_width=True):
            from generate_db import main as _gen
            with st.spinner("Generando base de datos…"):
                _gen()
            st.rerun()
        return

    if st.session_state.graph is None:
        with st.spinner("Construyendo agente SQL…"):
            _load_sql_agent(str(db_path))
        st.rerun()

    st.success("Base de datos cargada")
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    st.markdown(f"**Tablas ({len(tables)}):**")
    for (name,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        st.markdown(f"&nbsp;&nbsp;• `{name}` — {count:,} filas")
    conn.close()


# ---------------------------------------------------------------------------
# Sidebar principal
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 Agente Analista")

    agent_choice = st.radio(
        "Tipo de agente",
        ["📊 Excel", "🗄️ SQL"],
        horizontal=True,
    )
    selected_type = "excel" if "Excel" in agent_choice else "sql"

    # Reset al cambiar de agente
    if selected_type != st.session_state.agent_type:
        for k in ["graph", "messages", "excel_path", "df",
                  "all_sheets", "db_path", "_file_key"]:
            st.session_state.pop(k, None)
        st.session_state.agent_type = selected_type
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    if st.session_state.agent_type == "excel":
        _render_excel_sidebar()
    else:
        _render_sql_sidebar()

    st.markdown("---")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    st.markdown(f"**Modelo:** `{model_name}`")
    if st.button("🔄 Nueva sesión", use_container_width=True):
        _reset_session()


# ---------------------------------------------------------------------------
# Área principal — pantalla de bienvenida
# ---------------------------------------------------------------------------

WELCOME = {
    "excel": """
# Agente Analista de Excel 📊
Sube un archivo Excel en el panel lateral para empezar.

**Herramientas disponibles:**
- `get_dataframe_info` — schema, estadísticas, filas de muestra
- `list_sheets` — hojas del workbook
- `python_repl` — código pandas arbitrario
- `generate_chart` — gráficas matplotlib inline
""",
    "sql": """
# Agente Analista SQL 🗄️
La base de datos se carga automáticamente desde `data/sample.db`.

**Herramientas disponibles:**
- `list_tables` — tablas disponibles
- `describe_table` — schema y muestra de una tabla
- `run_sql` — consultas SELECT con JOIN entre tablas
- `generate_chart` — gráficas matplotlib inline
""",
}

if st.session_state.graph is None:
    st.markdown(WELCOME[st.session_state.agent_type])
    st.stop()

# Cabecera dinámica según tipo de agente
if st.session_state.agent_type == "excel":
    excel_name = Path(st.session_state.excel_path).name
    df = st.session_state.df
    st.markdown(f"### 📁 `{excel_name}` &nbsp;—&nbsp; {len(df):,} filas × {len(df.columns)} columnas")
else:
    db_name = Path(st.session_state.db_path).name
    st.markdown(f"### 🗄️ `{db_name}`")

st.markdown("---")

# ---------------------------------------------------------------------------
# Historial de chat (idéntico para ambos agentes)
# ---------------------------------------------------------------------------

for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        for path in entry.get("charts", []):
            st.image(path, use_container_width=True)
        if entry.get("tool_calls"):
            with st.expander("🔧 Llamadas a herramientas", expanded=False):
                for tc in entry["tool_calls"]:
                    st.markdown(f"**`{tc['name']}`**")
                    if tc["input"]:
                        st.json(tc["input"])
                    if tc["output"] and not tc["output"].startswith(CHART_PREFIX):
                        st.code(tc["output"], language="text")

# ---------------------------------------------------------------------------
# Input de chat (idéntico para ambos agentes)
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Haz una pregunta o pide una gráfica…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        with st.spinner("El agente está pensando…"):
            try:
                result = st.session_state.graph.invoke(
                    {"messages": [("human", prompt)]},
                    config=config,
                )
                answer = extract_final_answer(result)
                tool_calls = _extract_tool_calls(result)
            except Exception as exc:
                answer = f"**Error del agente:** {exc}"
                tool_calls = []

        st.markdown(answer)

        chart_paths = _extract_chart_paths(tool_calls)
        for path in chart_paths:
            st.image(path, use_container_width=True)

        if tool_calls:
            with st.expander("🔧 Llamadas a herramientas", expanded=False):
                for tc in tool_calls:
                    st.markdown(f"**`{tc['name']}`**")
                    if tc["input"]:
                        st.json(tc["input"])
                    if tc["output"] and not tc["output"].startswith(CHART_PREFIX):
                        st.code(tc["output"], language="text")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_calls": tool_calls,
        "charts": chart_paths,
    })
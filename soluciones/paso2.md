# Solución Paso 2 — UI web con Streamlit

Partimos de la solución del Paso 1 (agente terminal) y añadimos una interfaz web con **Streamlit** sin tocar nada de `agent.py`, `tools.py` ni `state.py`.

Duración estimada: 45-60 min.

---

## Índice

1. [¿Qué añadimos en este paso?](#1-qué-añadimos-en-este-paso)
2. [Concepto clave: Streamlit y session_state](#2-concepto-clave-streamlit-y-session_state)
3. [Instalar dependencia](#3-instalar-dependencia)
4. [app.py — La UI completa](#4-apppy--la-ui-completa)
5. [Ejecución](#5-ejecución)
6. [Preguntas de ejemplo](#6-preguntas-de-ejemplo)
7. [Puntos de discusión](#7-puntos-de-discusión)

---

## 1. ¿Qué añadimos en este paso?

Solo un archivo nuevo: **`app.py`**.

```
chat.py      ← (paso 1) CLI terminal — no cambia
app.py       ← (paso 2) UI web Streamlit — NUEVO
agent.py     ← no cambia
tools.py     ← no cambia
state.py     ← no cambia
```

El punto clave: **Streamlit es solo una capa de UI**. El agente, las herramientas y el estado son exactamente los mismos que en el paso 1. Esto demuestra que separar la lógica de negocio de la presentación facilita reutilizar código.

---

## 2. Concepto clave: Streamlit y session_state

Streamlit **re-ejecuta el script completo** en cada interacción del usuario (cada clic, cada mensaje). Esto significa que las variables locales se reinician en cada rerun.

Para preservar estado entre reruns usamos `st.session_state`, que es un diccionario que persiste durante toda la sesión del navegador.

```python
# MAL: se resetea en cada rerun
graph = build_agent(tools)   # ← se reconstruye en cada mensaje

# BIEN: se construye solo una vez
if "graph" not in st.session_state:
    st.session_state.graph = build_agent(tools)
```

Las claves que necesitamos persistir:

| Clave | Tipo | Para qué |
|-------|------|----------|
| `graph` | `CompiledGraph` | El agente LangGraph ya construido |
| `thread_id` | `str` | UUID fijo por sesión para MemorySaver |
| `messages` | `list[dict]` | Historial de chat para renderizar |
| `df` | `pd.DataFrame` | DataFrame principal cargado |
| `all_sheets` | `dict` | Todas las hojas del workbook |
| `excel_path` | `str` | Ruta al archivo temporal |

---

## 3. Instalar dependencia

```bash
pip install streamlit>=1.35.0
```

O añade al `requirements.txt`:

```
streamlit>=1.35.0
```

---

## 4. app.py — La UI completa

```python
# app.py — Interfaz web Streamlit para el agente analista de Excel.
#
# Uso:
#     streamlit run app.py
#
# Teaching points:
#   - Streamlit re-ejecuta el script entero en cada interacción.
#     session_state es el escape hatch para todo lo que debe persistir.
#   - Reutilizamos agent.py, tools.py y state.py sin modificarlos.
#     Streamlit es solo una capa de UI.
#   - thread_id fijo por sesión → memoria multi-turno automática.
#   - Las llamadas a herramientas se muestran en un expander para que
#     el usuario vea el razonamiento del agente (transparencia).

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

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración de página (debe ser la primera llamada a Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agente Analista de Excel",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Inicialización de session_state
# ---------------------------------------------------------------------------
# Estas claves se establecen una sola vez; los reruns posteriores
# saltan los bloques `not in`.

if "graph" not in st.session_state:
    st.session_state.graph = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []        # list[dict] — {role, content, tool_calls}
if "excel_path" not in st.session_state:
    st.session_state.excel_path = None
if "df" not in st.session_state:
    st.session_state.df = None
if "all_sheets" not in st.session_state:
    st.session_state.all_sheets = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_build(uploaded_file) -> None:
    """
    Guarda el archivo subido en un fichero temporal, lo carga en un
    DataFrame, construye las herramientas y el agente, y almacena todo
    en session_state.

    Por qué un fichero temporal: pandas/openpyxl necesitan un archivo
    real en disco (con seek) para leer .xlsx correctamente.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
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
    # Resetear conversación al cargar un archivo nuevo
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())


def _reset_session() -> None:
    """Limpia todo el session_state (botón 'Nueva sesión')."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def extract_final_answer(result: dict) -> str:
    """Extrae el último mensaje AI con texto de la respuesta del agente."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            if not getattr(msg, "tool_calls", None):
                return msg.content
    return "(Sin respuesta generada)"


def _extract_tool_calls(result: dict) -> list[dict]:
    """
    Devuelve lista de {name, input, output} de las llamadas a herramientas.
    Se usa para poblar el expander de Tool calls.
    """
    calls = []
    messages = result.get("messages", [])

    # Índice tool_call_id → output del ToolMessage
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
    st.title("📊 Agente Excel")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Sube un archivo Excel",
        type=["xlsx", "xls"],
        help="Formatos soportados: .xlsx, .xls",
    )

    if uploaded is not None:
        # Reconstruir solo cuando se sube un archivo *nuevo* (por nombre + tamaño)
        file_key = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_file_key") != file_key:
            with st.spinner("Cargando archivo y construyendo agente…"):
                _load_and_build(uploaded)
            st.session_state["_file_key"] = file_key
            st.success("¡Agente listo!")

    st.markdown("---")

    if st.button("🔄 Nueva sesión", use_container_width=True):
        _reset_session()

    # Info del archivo cargado
    if st.session_state.df is not None:
        df = st.session_state.df
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        st.markdown(f"**Modelo:** `{model_name}`")
        st.markdown(f"**Filas:** {len(df):,}  |  **Columnas:** {len(df.columns)}")
        sheet_names = list(st.session_state.all_sheets.keys())
        st.markdown(f"**Hojas ({len(sheet_names)}):**")
        for name in sheet_names:
            n_rows = len(st.session_state.all_sheets[name])
            st.markdown(f"&nbsp;&nbsp;• `{name}` — {n_rows:,} filas")


# ---------------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------------

if st.session_state.graph is None:
    # Pantalla de bienvenida antes de subir un archivo
    st.markdown(
        """
        # Bienvenido al Agente Analista de Excel 📊

        Sube un archivo Excel en el panel lateral para empezar.

        **Qué puedes preguntar:**
        - *"¿Qué columnas tiene este archivo?"*
        - *"Muéstrame los 5 productos con más revenue"*
        - *"¿Cuál es el valor medio de pedido por región?"*
        - *"¿Qué segmento de clientes genera más ingresos?"*

        El agente usa un bucle LangGraph ReAct con tres herramientas:
        - **get_dataframe_info** — schema, estadísticas, filas de muestra
        - **list_sheets** — nombres de hojas del workbook
        - **python_repl** — código pandas arbitrario via parsing AST

        La memoria multi-turno se mantiene durante toda la sesión.
        """
    )
    st.stop()

# Cabecera con info del archivo
excel_name = Path(st.session_state.excel_path).name if st.session_state.excel_path else "unknown"
df = st.session_state.df
st.markdown(f"### 📁 `{excel_name}` &nbsp;—&nbsp; {len(df):,} filas × {len(df.columns)} columnas")
st.markdown("---")

# ---------------------------------------------------------------------------
# Historial de chat
# ---------------------------------------------------------------------------

for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("tool_calls"):
            with st.expander("🔧 Llamadas a herramientas", expanded=False):
                for tc in entry["tool_calls"]:
                    st.markdown(f"**`{tc['name']}`**")
                    if tc["input"]:
                        st.json(tc["input"])
                    if tc["output"]:
                        st.code(tc["output"], language="text")

# ---------------------------------------------------------------------------
# Input de chat
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Haz una pregunta sobre tus datos…"):
    # Mostrar el mensaje del usuario inmediatamente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invocar el agente
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
        if tool_calls:
            with st.expander("🔧 Llamadas a herramientas", expanded=False):
                for tc in tool_calls:
                    st.markdown(f"**`{tc['name']}`**")
                    if tc["input"]:
                        st.json(tc["input"])
                    if tc["output"]:
                        st.code(tc["output"], language="text")

    # Guardar en el historial
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_calls": tool_calls,
    })
```

---

## 5. Ejecución

```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit abre automáticamente `http://localhost:8501` en el navegador.

---

## 6. Preguntas de ejemplo

Las mismas que en el paso 1 funcionan igual. Además, al tener interfaz gráfica puedes combinar varias preguntas seguidas y ver el historial completo de la conversación con el razonamiento del agente desplegable en cada respuesta.

---

## 7. Puntos de discusión

1. **¿Por qué `session_state` y no una variable global?**
   Streamlit puede tener múltiples usuarios concurrentes. Las variables globales se compartirían entre sesiones. `session_state` es por usuario/pestaña.

2. **¿Por qué `tempfile.NamedTemporaryFile`?**
   El objeto `UploadedFile` de Streamlit es un buffer en memoria. `openpyxl` necesita un archivo real en disco para hacer seeks. La solución es volcarlo a un archivo temporal.

3. **¿Por qué detectar el archivo por `nombre + tamaño` y no llamar siempre a `_load_and_build`?**
   Streamlit re-ejecuta el script en cada mensaje del chat. Sin ese guard, reconstruiríamos el agente (y perderíamos la memoria) en cada respuesta.

4. **¿Cómo añadirías streaming de tokens en Streamlit?**
   Sustituir `graph.invoke()` por `graph.stream()` y usar `st.write_stream()` para mostrar los tokens a medida que llegan.

# Solución Paso 4 — Agente SQL + selector de agente en la UI

Partimos del Paso 3 y añadimos un segundo agente que habla con una base de datos SQLite en lugar de un Excel. La UI permite cambiar entre ambos agentes sin recargar la página.

Duración estimada: 60-90 min.

---

## Índice

1. [¿Qué añadimos en este paso?](#1-qué-añadimos-en-este-paso)
2. [Diseño: dos agentes, una UI](#2-diseño-dos-agentes-una-ui)
3. [generate_db.py — crear el SQLite de ejemplo](#3-generate_dbpy--crear-el-sqlite-de-ejemplo)
4. [sql_tools.py — herramientas SQL](#4-sql_toolspy--herramientas-sql)
5. [sql_agent.py — el agente SQL](#5-sql_agentpy--el-agente-sql)
6. [app.py — selector de agente](#6-apppy--selector-de-agente)
7. [Ejecución y pruebas](#7-ejecución-y-pruebas)
8. [Puntos de discusión](#8-puntos-de-discusión)

---

## 1. ¿Qué añadimos en este paso?

| Archivo | Cambio |
|---------|--------|
| `generate_db.py` | Nuevo — genera `data/sample.db` desde los mismos datos |
| `sql_tools.py` | Nuevo — `list_tables`, `describe_table`, `run_sql`, `generate_chart` |
| `sql_agent.py` | Nuevo — ReAct agent con system prompt SQL |
| `app.py` | Modificado — selector de agente en la sidebar |
| `.gitignore` | Modificado — ignorar `data/*.db` |

`agent.py`, `tools.py`, `state.py` y `chat.py` **no se tocan**.

---

## 2. Diseño: dos agentes, una UI

El reto principal es compartir la infraestructura de la UI (historial, renderizado de gráficas, loop de chat) entre dos agentes completamente distintos.

La solución: **un único `graph` activo en `session_state`**. Cuando el usuario cambia de agente, reseteamos el estado y construimos un nuevo grafo.

```
session_state.agent_type = "excel" | "sql"
session_state.graph       = CompiledGraph activo (Excel o SQL)
session_state.messages    = historial de chat (se resetea al cambiar)
```

El bucle de chat (`graph.invoke → extract_answer → render`) es **idéntico** para ambos agentes. Solo cambia cómo se inicializa el grafo.

```
Sidebar Excel:  file_uploader → build_tools()    → build_agent()
Sidebar SQL:    auto-carga db → build_sql_tools() → build_sql_agent()
                                                      ↓
                              graph.invoke() — mismo código para los dos
```

Cuando se cambia de agente:
1. Se vacían las claves de session_state del agente anterior
2. Se establece `agent_type` al nuevo valor
3. `st.rerun()` redibuja la UI con el sidebar correcto

---

## 3. generate_db.py — crear el SQLite de ejemplo

Importamos las funciones generadoras de `generate_sample.py` y escribimos en SQLite.
Esto muestra reutilización de código: los datos son los mismos, solo cambia el destino.

```python
# generate_db.py
from pathlib import Path
import sqlite3
import pandas as pd

from generate_sample import make_products_df, make_clients_df, make_sales_df

DB_PATH = Path("data/sample.db")


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Generando datos de ejemplo...")
    products = make_products_df()
    clients = make_clients_df()
    sales = make_sales_df(clients, products)

    # SQLite no tiene tipo DATE nativo; guardamos como string ISO
    sales["date"] = sales["date"].dt.strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    products.to_sql("products", conn, if_exists="replace", index=False)
    clients.to_sql("clients", conn, if_exists="replace", index=False)
    sales.to_sql("sales", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Creada: {DB_PATH.resolve()}")
    print(f"  sales:    {len(sales)} filas")
    print(f"  clients:  {len(clients)} filas")
    print(f"  products: {len(products)} filas")


if __name__ == "__main__":
    main()
```

```bash
python generate_db.py
```

---

## 4. sql_tools.py — herramientas SQL

Cuatro tools que el agente SQL tiene disponibles:

| Tool | Cuándo usarla |
|------|---------------|
| `list_tables` | Primera llamada para explorar la BD |
| `describe_table` | Antes de escribir SQL: schema + muestra |
| `run_sql` | Ejecutar consultas SELECT |
| `generate_chart` | Igual que en paso 3 pero con `conn` en lugar de `df` |

**Puntos clave de diseño:**
- `check_same_thread=False` en `sqlite3.connect()`: necesario porque Streamlit puede ejecutar callbacks en hilos distintos.
- `run_sql` solo acepta SELECT: validación básica de seguridad con `query.strip().upper().startswith("SELECT")`.
- `generate_chart` inyecta `conn` (en lugar de `df`), así el LLM puede `pd.read_sql(query, conn)` para cargar datos y luego graficarlos.

```python
# sql_tools.py
import sqlite3
import tempfile
from typing import Any

import pandas as pd
from langchain.tools import tool


def build_sql_tools(db_path: str) -> list[Any]:
    # check_same_thread=False: Streamlit puede invocar tools desde hilos distintos
    conn = sqlite3.connect(db_path, check_same_thread=False)

    @tool
    def list_tables(dummy: str = "") -> str:
        """
        Lista todas las tablas disponibles en la base de datos SQLite.

        Usa esta herramienta PRIMERO para explorar la estructura de la BD
        antes de escribir consultas SQL.
        """
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        if not tables:
            return "No hay tablas en la base de datos."
        return "Tablas disponibles:\n" + "\n".join(f"  - {t}" for t in tables)

    @tool
    def describe_table(table_name: str) -> str:
        """
        Describe el schema de una tabla: columnas, tipos, nulos y filas de muestra.

        Args:
            table_name: Nombre de la tabla (sales, clients, products)
        """
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        cols = cursor.fetchall()
        if not cols:
            available = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            return (
                f"Tabla '{table_name}' no encontrada. "
                f"Disponibles: {[r[0] for r in available]}"
            )

        col_lines = [
            f"  {c[1]:20s} {c[2]:15s}" + (" NOT NULL" if c[3] else "")
            for c in cols
        ]
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        sample = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 3", conn)

        return (
            f"Tabla: {table_name} ({count:,} filas)\n"
            "Columnas:\n" + "\n".join(col_lines) +
            f"\n\nPrimeras 3 filas:\n{sample.to_markdown(index=False)}"
        )

    @tool
    def run_sql(query: str) -> str:
        """
        Ejecuta una consulta SQL SELECT y devuelve los resultados en formato tabla.

        Solo se permiten SELECT. Para consultas cruzadas usa JOIN:
            SELECT s.region, SUM(s.total_revenue) as revenue
            FROM sales s
            JOIN clients c ON s.client_id = c.client_id
            GROUP BY s.region
            ORDER BY revenue DESC

        Args:
            query: Consulta SQL SELECT válida.
        """
        if not query.strip().upper().startswith("SELECT"):
            return "Error: solo se permiten consultas SELECT."
        try:
            df = pd.read_sql(query, conn)
            if df.empty:
                return "La consulta no devolvió resultados."
            return df.to_markdown(index=False)
        except Exception as exc:
            return f"Error SQL: {exc}"

    @tool
    def generate_chart(code: str) -> str:
        """
        Genera una gráfica ejecutando código Python y la guarda como PNG.

        IMPORTANTE: NO llames a plt.show(). La figura se guarda automáticamente.

        Args:
            code: Código Python que crea una figura matplotlib.

        Variables disponibles:
            conn — conexión sqlite3 (usa pd.read_sql(query, conn) para datos)
            pd   — pandas
            plt  — matplotlib.pyplot

        Ejemplos:

            # Revenue por región
            df = pd.read_sql(
                "SELECT region, SUM(total_revenue) as revenue FROM sales GROUP BY region ORDER BY revenue",
                conn
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            df.plot(kind='barh', x='region', y='revenue', ax=ax, legend=False)
            ax.set_title('Revenue por región')

            # Ventas mensuales
            df = pd.read_sql(
                "SELECT strftime('%Y-%m', date) as mes, SUM(total_revenue) as revenue FROM sales GROUP BY mes ORDER BY mes",
                conn
            )
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['mes'], df['revenue'], marker='o')
            ax.set_title('Revenue mensual 2024')
            plt.xticks(rotation=45)

            # JOIN entre tablas
            df = pd.read_sql(
                "SELECT c.segment, SUM(s.total_revenue) as revenue FROM sales s JOIN clients c ON s.client_id = c.client_id GROUP BY c.segment",
                conn
            )
            fig, ax = plt.subplots()
            df.plot(kind='pie', y='revenue', labels=df['segment'], autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            ax.set_title('Revenue por segmento')

        Returns:
            "CHART:/ruta/al/archivo.png" si tiene éxito, o mensaje de error.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        local_ns = {"conn": conn, "pd": pd, "plt": plt}

        try:
            exec(compile(code, "<chart>", "exec"), local_ns)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="chart_")
            plt.savefig(tmp.name, bbox_inches="tight", dpi=150)
            plt.close("all")
            return f"CHART:{tmp.name}"
        except Exception as exc:
            plt.close("all")
            return f"Error al generar la gráfica: {exc}"

    return [list_tables, describe_table, run_sql, generate_chart]
```

---

## 5. sql_agent.py — el agente SQL

Idéntico a `agent.py` en estructura; solo cambia el system prompt para orientar al LLM hacia SQL en lugar de pandas.

```python
# sql_agent.py
import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """Eres un analista de datos experto en SQL y SQLite.
Tu función es ayudar a los usuarios a explorar y entender bases de datos mediante lenguaje natural.

REGLAS:
1. Llama siempre a `list_tables` primero si no conoces la estructura de la BD.
2. Usa `describe_table` para explorar columnas y tipos antes de escribir SQL.
3. Solo SELECT: nunca escribas INSERT, UPDATE, DELETE ni DDL.
4. Explica los resultados en lenguaje llano tras cada consulta.
5. Para gráficas usa `generate_chart` con pd.read_sql(query, conn) para cargar datos.
6. Formatea los números con claridad: miles con comas, 2 decimales.

TABLAS DISPONIBLES:
- sales    — transacciones (sale_id, date, quarter, client_id, product_id, region, segment, total_revenue, salesperson...)
- clients  — clientes (client_id, company, region, segment, since_year)
- products — productos (product_id, product_name, category, unit_price, stock_units, supplier)

Relaciones: sales.client_id → clients.client_id | sales.product_id → products.product_id

Para consultas cruzadas usa JOIN:
    SELECT c.segment, SUM(s.total_revenue)
    FROM sales s JOIN clients c ON s.client_id = c.client_id
    GROUP BY c.segment

Responde siempre en el idioma del usuario."""


def build_sql_agent(tools: list):
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    memory = MemorySaver()
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )
```

---

## 6. app.py — selector de agente

### 6a. Nuevo session_state

```python
if "agent_type" not in st.session_state:
    st.session_state.agent_type = "excel"   # "excel" | "sql"
if "db_path" not in st.session_state:
    st.session_state.db_path = None
```

### 6b. Selector en la sidebar + reset al cambiar

```python
with st.sidebar:
    st.title("🤖 Agente Analista")

    agent_choice = st.radio("Tipo de agente", ["📊 Excel", "🗄️ SQL"], horizontal=True)
    selected_type = "excel" if "Excel" in agent_choice else "sql"

    # Al cambiar de agente, limpiar estado del anterior y recargar
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
```

### 6c. Sidebar SQL: auto-carga si existe el .db

```python
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

    # Auto-cargar si el agente no está construido
    if st.session_state.graph is None:
        with st.spinner("Construyendo agente SQL…"):
            _load_sql_agent(str(db_path))
        st.rerun()

    # Info
    st.success("Base de datos cargada")
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    st.markdown(f"**Tablas ({len(tables)}):**")
    for (name,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        st.markdown(f"&nbsp;&nbsp;• `{name}` — {count:,} filas")
    conn.close()
```

### Código completo de app.py

```python
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
```

---

## 7. Ejecución y pruebas

```bash
source .venv/bin/activate

# Generar la BD (solo la primera vez)
python generate_db.py

# Arrancar la UI
streamlit run app.py
```

**Seleccionar agente SQL** en el radio de la sidebar → la BD se carga automáticamente.

**Preguntas de prueba para el agente SQL:**

```sql
-- Análisis simple
¿Cuántas ventas hay por región?
¿Cuál es el producto más vendido por unidades?
¿Qué trimestre tuvo más revenue?

-- Consultas con JOIN
¿Cuánto revenue genera cada segmento de cliente (SMB, Enterprise, Public Sector)?
¿Qué vendedor tiene el ticket medio más alto?
¿Qué proveedor suministra los productos más vendidos?

-- Gráficas
Muéstrame un gráfico de barras del revenue mensual
Genera un pie chart del revenue por categoría de producto
Crea un gráfico comparando el revenue por región y segmento
```

---

## 8. Puntos de discusión

1. **¿Por qué `check_same_thread=False` en la conexión SQLite?**
   Streamlit puede ejecutar callbacks en hilos distintos al principal. SQLite por defecto lanza `ProgrammingError: SQLite objects created in a thread can only be used in that same thread`. Con `check_same_thread=False` permitimos el acceso desde cualquier hilo (seguro para lectura, que es todo lo que hacemos).

2. **¿Por qué el agente SQL no usa LangChain's `SQLDatabaseToolkit`?**
   `SQLDatabaseToolkit` es conveniente para prod, pero oculta cómo funcionan las tools. Al construirlas desde cero con `@tool` los alumnos ven exactamente qué recibe el LLM, cómo se valida el SQL y cómo se pasa la conexión. Esto facilita adaptar las tools a cualquier BD.

3. **¿Cómo funciona el reset al cambiar de agente?**
   El `st.radio` devuelve el valor seleccionado en cada rerun. Comparamos con `session_state.agent_type`. Si son distintos, limpiamos las claves del agente anterior y llamamos a `st.rerun()`. Streamlit redibuja el sidebar con el nuevo agente y la pantalla de bienvenida correspondiente.

4. **¿Por qué el bucle de chat es el mismo para ambos agentes?**
   Porque la interfaz del grafo es idéntica: `graph.invoke({"messages": [...]}, config)` → `{"messages": [...]}`. El patrón ReAct está encapsulado en `create_react_agent`; el código de la UI no necesita saber qué tools tiene el agente activo.

5. **¿Cómo añadirías un tercer agente (p.ej., para APIs REST)?**
   1. Crear `api_tools.py` y `api_agent.py`
   2. Añadir `"🌐 API"` al radio
   3. Añadir `_render_api_sidebar()` y `_load_api_agent()`
   4. El bucle de chat no cambia.

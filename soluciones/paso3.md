# Solución Paso 3 — Generación de gráficas

Partimos del Paso 2 (agente + UI Streamlit) y añadimos la capacidad de generar gráficas matplotlib directamente desde el chat.

Duración estimada: 45-60 min.

---

## Índice

1. [¿Qué añadimos en este paso?](#1-qué-añadimos-en-este-paso)
2. [Diseño: cómo pasar una imagen del agente a la UI](#2-diseño-cómo-pasar-una-imagen-del-agente-a-la-ui)
3. [tools.py — nueva tool `generate_chart`](#3-toolspy--nueva-tool-generate_chart)
4. [app.py — mostrar gráficas inline en el chat](#4-apppy--mostrar-gráficas-inline-en-el-chat)
5. [chat.py — informar y abrir la gráfica en terminal](#5-chatpy--informar-y-abrir-la-gráfica-en-terminal)
6. [requirements.txt — añadir matplotlib](#6-requirementstxt--añadir-matplotlib)
7. [Ejecución y pruebas](#7-ejecución-y-pruebas)
8. [Puntos de discusión](#8-puntos-de-discusión)

---

## 1. ¿Qué añadimos en este paso?

| Archivo | Cambio |
|---------|--------|
| `tools.py` | Nueva tool `generate_chart` |
| `app.py` | Detectar y mostrar imágenes inline en el chat |
| `chat.py` | Detectar y abrir/informar gráficas en terminal |
| `requirements.txt` | Añadir `matplotlib` |

`agent.py` y `state.py` **no se tocan**.

---

## 2. Diseño: cómo pasar una imagen del agente a la UI

El problema: el agente se ejecuta dentro de `graph.invoke()`, sin acceso directo al contexto de Streamlit ni al terminal. Las tools solo pueden devolver **strings**.

La solución: protocolo de señal con prefijo.

```
generate_chart devuelve → "CHART:/tmp/chart_abc123.png"
```

Después de `graph.invoke()`, escaneamos los outputs de las tool calls buscando el prefijo `CHART:`. Si lo encontramos, renderizamos la imagen.

```
Tool output: "CHART:/tmp/chart_abc123.png"
                 ↓
app.py detecta el prefijo → st.image("/tmp/chart_abc123.png")
chat.py detecta el prefijo → subprocess.run(["open", path])  # macOS
```

Este patrón es simple, explícito y funciona sin modificar la infraestructura del agente.

---

## 3. tools.py — nueva tool `generate_chart`

Solo hay que añadir la nueva tool dentro de `build_tools()` e incluirla en el `return`.

```python
# tools.py  (añadir al inicio del archivo)
import subprocess
import tempfile

# ... (resto de imports y código existente sin cambios) ...

def build_tools(df: pd.DataFrame, excel_path: str) -> list[Any]:
    # ... (código existente sin cambios hasta el final) ...

    # ------------------------------------------------------------------
    # Herramienta 4: generate_chart
    # ------------------------------------------------------------------
    @tool
    def generate_chart(code: str) -> str:
        """
        Genera una gráfica ejecutando código matplotlib y la guarda como PNG.

        IMPORTANTE: NO llames a plt.show(). La figura se guarda automáticamente.

        Args:
            code: Código Python que crea una figura matplotlib.

        Variables disponibles:
            df          — DataFrame principal
            all_sheets  — dict {nombre_hoja: DataFrame}
            pd          — pandas
            plt         — matplotlib.pyplot

        Ejemplos de uso:

            # Gráfico de barras por región
            fig, ax = plt.subplots(figsize=(10, 6))
            df.groupby('region')['total_revenue'].sum().sort_values().plot(kind='barh', ax=ax)
            ax.set_title('Revenue por región')

            # Serie temporal mensual
            fig, ax = plt.subplots(figsize=(12, 5))
            df.set_index('date')['total_revenue'].resample('ME').sum().plot(ax=ax)
            ax.set_title('Revenue mensual 2024')

            # Combinar hojas con merge
            fig, ax = plt.subplots(figsize=(10, 6))
            merged = pd.merge(all_sheets['Sales'], all_sheets['Clients'], on='client_id')
            merged.groupby('segment')['total_revenue'].sum().plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title('Revenue por segmento de cliente')

        Returns:
            "CHART:/ruta/al/archivo.png" si tiene éxito, o mensaje de error.
        """
        import matplotlib
        matplotlib.use("Agg")   # Backend sin ventana, necesario fuera de entorno gráfico
        import matplotlib.pyplot as plt

        local_ns = {
            "df": df,
            "all_sheets": all_sheets,
            "pd": pd,
            "plt": plt,
        }

        try:
            exec(compile(code, "<chart>", "exec"), local_ns)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="chart_")
            plt.savefig(tmp.name, bbox_inches="tight", dpi=150)
            plt.close("all")
            return f"CHART:{tmp.name}"
        except Exception as exc:
            plt.close("all")
            return f"Error al generar la gráfica: {exc}"

    return [get_dataframe_info, list_sheets, repl, generate_chart]
```

### Código completo de tools.py

```python
# tools.py
import io
import tempfile
from typing import Any

import pandas as pd
from langchain.tools import tool
from langchain_experimental.tools import PythonAstREPLTool


def _capture_df_info(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def build_tools(df: pd.DataFrame, excel_path: str) -> list[Any]:
    all_sheets: dict[str, pd.DataFrame] = pd.read_excel(
        excel_path, sheet_name=None, engine=_pick_engine(excel_path)
    )

    @tool
    def get_dataframe_info(dummy: str = "") -> str:
        """
        Devuelve el schema y datos de muestra del DataFrame activo (`df`).

        Usa esta herramienta PRIMERO cuando el usuario haga una pregunta sobre
        los datos, antes de escribir código pandas. Proporciona:
          - Nombres de columna y tipos de dato (df.info())
          - Estadísticas descriptivas (df.describe())
          - Primeras 5 filas (df.head())

        El parámetro `dummy` no se usa; pasa una cadena vacía o cualquier valor.
        """
        info = _capture_df_info(df)
        describe = df.describe(include="all").to_string()
        head = df.head(5).to_markdown(index=False)
        return (
            f"=== df.info() ===\n{info}\n"
            f"=== df.describe() ===\n{describe}\n\n"
            f"=== df.head(5) ===\n{head}"
        )

    @tool
    def list_sheets(dummy: str = "") -> str:
        """
        Lista todos los nombres de hojas disponibles en el workbook Excel.

        Usa esta herramienta cuando el usuario pregunte sobre la estructura del
        archivo, quiera saber cuántas hojas existen, o mencione una hoja por nombre.

        En el python_repl, todas las hojas son accesibles via el dict `all_sheets`:
            all_sheets['NombreHoja']  ->  pd.DataFrame

        Para combinar hojas usa pd.merge():
            pd.merge(all_sheets['Sales'], all_sheets['Clients'], on='client_id')

        El parámetro `dummy` no se usa; pasa una cadena vacía o cualquier valor.
        """
        names = list(all_sheets.keys())
        lines = [
            f"  {i + 1}. '{name}' — {len(all_sheets[name])} filas"
            for i, name in enumerate(names)
        ]
        return "Hojas disponibles en el workbook:\n" + "\n".join(lines)

    repl = PythonAstREPLTool(
        locals={"df": df, "all_sheets": all_sheets, "pd": pd},
        description=(
            "Ejecuta código Python / pandas para analizar los datos del Excel.\n"
            "Variables disponibles:\n"
            "  df          — DataFrame principal (solo lectura, NO reasignar)\n"
            "  all_sheets  — dict que mapea nombre de hoja -> DataFrame\n"
            "  pd          — módulo pandas\n\n"
            "Reglas:\n"
            "  - NUNCA modificar df en su lugar (no df.drop(inplace=True), etc.)\n"
            "  - Usa .to_markdown() para output de tablas legible\n"
            "  - Mantén el código conciso; una expresión por paso de análisis\n\n"
            "Ejemplos:\n"
            "  df.groupby('category')['total_revenue'].sum().sort_values(ascending=False)\n"
            "  pd.merge(all_sheets['Sales'], all_sheets['Clients'], on='client_id')"
            ".groupby('segment')['total_revenue'].sum()\n"
            "  all_sheets['Products'].sort_values('unit_price', ascending=False).head(5)"
        ),
    )

    @tool
    def generate_chart(code: str) -> str:
        """
        Genera una gráfica ejecutando código matplotlib y la guarda como PNG.

        IMPORTANTE: NO llames a plt.show(). La figura se guarda automáticamente.

        Args:
            code: Código Python que crea una figura matplotlib.

        Variables disponibles:
            df          — DataFrame principal
            all_sheets  — dict {nombre_hoja: DataFrame}
            pd          — pandas
            plt         — matplotlib.pyplot

        Ejemplos de uso:

            # Gráfico de barras por región
            fig, ax = plt.subplots(figsize=(10, 6))
            df.groupby('region')['total_revenue'].sum().sort_values().plot(kind='barh', ax=ax)
            ax.set_title('Revenue por región')

            # Serie temporal mensual
            fig, ax = plt.subplots(figsize=(12, 5))
            df.set_index('date')['total_revenue'].resample('ME').sum().plot(ax=ax)
            ax.set_title('Revenue mensual 2024')

            # Combinar hojas con merge
            fig, ax = plt.subplots(figsize=(10, 6))
            merged = pd.merge(all_sheets['Sales'], all_sheets['Clients'], on='client_id')
            merged.groupby('segment')['total_revenue'].sum().plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title('Revenue por segmento de cliente')

        Returns:
            "CHART:/ruta/al/archivo.png" si tiene éxito, o mensaje de error.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        local_ns = {
            "df": df,
            "all_sheets": all_sheets,
            "pd": pd,
            "plt": plt,
        }

        try:
            exec(compile(code, "<chart>", "exec"), local_ns)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="chart_")
            plt.savefig(tmp.name, bbox_inches="tight", dpi=150)
            plt.close("all")
            return f"CHART:{tmp.name}"
        except Exception as exc:
            plt.close("all")
            return f"Error al generar la gráfica: {exc}"

    return [get_dataframe_info, list_sheets, repl, generate_chart]


def _pick_engine(path: str) -> str:
    return "xlrd" if path.lower().endswith(".xls") else "openpyxl"
```

---

## 4. app.py — mostrar gráficas inline en el chat

Tres cambios respecto al paso 2:

### 4a. Extraer rutas de gráficas de los tool calls

Añadir helper `_extract_chart_paths()`:

```python
CHART_PREFIX = "CHART:"

def _extract_chart_paths(tool_calls: list[dict]) -> list[str]:
    """Extrae las rutas PNG de los outputs de generate_chart."""
    return [
        tc["output"][len(CHART_PREFIX):]
        for tc in tool_calls
        if isinstance(tc.get("output"), str) and tc["output"].startswith(CHART_PREFIX)
    ]
```

### 4b. Mostrar gráficas después de la respuesta de texto

```python
# En el bloque with st.chat_message("assistant"):
st.markdown(answer)
chart_paths = _extract_chart_paths(tool_calls)
for path in chart_paths:
    st.image(path, use_container_width=True)
```

### 4c. Guardar las rutas en el historial y re-renderizarlas

```python
# Al guardar el mensaje en session_state:
st.session_state.messages.append({
    "role": "assistant",
    "content": answer,
    "tool_calls": tool_calls,
    "charts": chart_paths,   # ← NUEVO
})

# En el bucle de historial:
for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        for path in entry.get("charts", []):        # ← NUEVO
            st.image(path, use_container_width=True)
        # expander de tool calls...
```

### Código completo de app.py

```python
# app.py
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

st.set_page_config(
    page_title="Agente Analista de Excel",
    page_icon="📊",
    layout="wide",
)

if "graph" not in st.session_state:
    st.session_state.graph = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "excel_path" not in st.session_state:
    st.session_state.excel_path = None
if "df" not in st.session_state:
    st.session_state.df = None
if "all_sheets" not in st.session_state:
    st.session_state.all_sheets = {}

CHART_PREFIX = "CHART:"


def _load_and_build(uploaded_file) -> None:
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
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())


def _reset_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def extract_final_answer(result: dict) -> str:
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            if not getattr(msg, "tool_calls", None):
                return msg.content
    return "(Sin respuesta generada)"


def _extract_tool_calls(result: dict) -> list[dict]:
    calls = []
    messages = result.get("messages", [])
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


def _extract_chart_paths(tool_calls: list[dict]) -> list[str]:
    """Extrae las rutas PNG de los outputs de generate_chart."""
    return [
        tc["output"][len(CHART_PREFIX):]
        for tc in tool_calls
        if isinstance(tc.get("output"), str) and tc["output"].startswith(CHART_PREFIX)
    ]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 Agente Excel")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Sube un archivo Excel",
        type=["xlsx", "xls"],
    )

    if uploaded is not None:
        file_key = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_file_key") != file_key:
            with st.spinner("Cargando archivo y construyendo agente…"):
                _load_and_build(uploaded)
            st.session_state["_file_key"] = file_key
            st.success("¡Agente listo!")

    st.markdown("---")

    if st.button("🔄 Nueva sesión", use_container_width=True):
        _reset_session()

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
    st.markdown(
        """
        # Bienvenido al Agente Analista de Excel 📊

        Sube un archivo Excel en el panel lateral para empezar.

        **Qué puedes pedir:**
        - *"¿Cuál es el revenue por región?"*
        - *"Muéstrame una gráfica de ventas mensuales"*
        - *"Genera un gráfico de barras por categoría de producto"*
        - *"¿Qué segmento genera más ingresos? Ponlo en un pie chart"*

        El agente dispone de cuatro herramientas:
        - **get_dataframe_info** — schema, estadísticas, filas de muestra
        - **list_sheets** — nombres de hojas del workbook
        - **python_repl** — código pandas arbitrario
        - **generate_chart** — gráficas matplotlib inline ← NUEVO
        """
    )
    st.stop()

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
        for path in entry.get("charts", []):            # mostrar gráficas guardadas
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
# Input de chat
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

## 5. chat.py — informar y abrir la gráfica en terminal

En el terminal no podemos renderizar imágenes inline. En su lugar:
1. Detectamos los `CHART:` en los tool call outputs
2. Mostramos la ruta al usuario
3. En macOS/Linux abrimos el visor de imágenes por defecto

```python
# Añadir helper en chat.py

import platform
import subprocess

CHART_PREFIX = "CHART:"

def _handle_charts(result: dict, console: Console) -> None:
    """Detecta gráficas generadas y las abre con el visor del sistema."""
    messages = result.get("messages", [])
    from langchain_core.messages import ToolMessage
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content.startswith(CHART_PREFIX):
            path = msg.content[len(CHART_PREFIX):]
            console.print(f"\n[bold yellow]📊 Gráfica guardada:[/] {path}")
            try:
                if platform.system() == "Darwin":
                    subprocess.run(["open", path], check=True)
                elif platform.system() == "Linux":
                    subprocess.run(["xdg-open", path], check=True)
            except Exception:
                pass   # Si no hay visor disponible, solo mostramos la ruta
```

Y en el bucle principal de `main()`, después de `graph.invoke()`:

```python
answer = extract_final_answer(result)
console.print()
console.print(Panel(Markdown(answer), title="[bold green]Agente[/]", border_style="green"))
_handle_charts(result, console)   # ← NUEVO
```

### Código completo de chat.py

```python
# chat.py
import platform
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from agent import build_agent
from tools import build_tools, _pick_engine

load_dotenv()

console = Console()
SUPPORTED_EXTENSIONS = {".xlsx", ".xls"}
CHART_PREFIX = "CHART:"


def load_excel(path: str) -> tuple[pd.DataFrame, str]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        console.print(f"[bold red]Error:[/] Archivo no encontrado: {p}")
        sys.exit(1)
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        console.print(f"[bold red]Error:[/] Formato no soportado '{p.suffix}'.")
        sys.exit(1)
    engine = _pick_engine(str(p))
    df = pd.read_excel(str(p), engine=engine)
    return df, str(p)


def print_welcome(excel_path: str, df: pd.DataFrame) -> None:
    info = (
        f"[bold cyan]Archivo:[/] {excel_path}\n"
        f"[bold cyan]Filas:[/] {len(df):,}   "
        f"[bold cyan]Columnas:[/] {len(df.columns)}\n"
        f"[bold cyan]Columnas:[/] {', '.join(df.columns.tolist())}\n\n"
        "[dim]Escribe tu pregunta o pide una gráfica. Escribe [bold]salir[/] para terminar.[/dim]"
    )
    console.print(Panel(info, title="[bold green]Agente Analista de Excel[/]", border_style="green"))


def extract_final_answer(result: dict) -> str:
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            if not getattr(msg, "tool_calls", None):
                return msg.content
    return "(Sin respuesta generada)"


def _handle_charts(result: dict) -> None:
    """Detecta gráficas generadas y las abre con el visor del sistema."""
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage) and msg.content.startswith(CHART_PREFIX):
            path = msg.content[len(CHART_PREFIX):]
            console.print(f"\n[bold yellow]📊 Gráfica guardada:[/] {path}")
            try:
                if platform.system() == "Darwin":
                    subprocess.run(["open", path], check=True)
                elif platform.system() == "Linux":
                    subprocess.run(["xdg-open", path], check=True)
            except Exception:
                pass


def main() -> None:
    if len(sys.argv) > 1:
        excel_path_arg = sys.argv[1]
    else:
        excel_path_arg = Prompt.ask("[bold yellow]Ruta al archivo Excel[/]")

    df, excel_path = load_excel(excel_path_arg)

    with console.status("[bold green]Cargando Excel y construyendo agente…[/]"):
        tools = build_tools(df, excel_path)
        graph = build_agent(tools)

    print_welcome(excel_path, df)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        console.print()
        try:
            user_input = Prompt.ask("[bold blue]Tú[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]¡Hasta luego![/dim]")
            break

        if user_input.strip().lower() in {"salir", "exit", "quit", "q"}:
            console.print("[dim]¡Hasta luego![/dim]")
            break

        if not user_input.strip():
            continue

        console.print(Rule("[dim]Agente pensando…[/dim]", style="dim"))
        try:
            result = graph.invoke(
                {"messages": [("human", user_input)]},
                config=config,
            )
        except Exception as exc:
            console.print(f"[bold red]Error del agente:[/] {exc}")
            continue

        answer = extract_final_answer(result)
        console.print()
        console.print(Panel(Markdown(answer), title="[bold green]Agente[/]", border_style="green"))
        _handle_charts(result)


if __name__ == "__main__":
    main()
```

---

## 6. requirements.txt — añadir matplotlib

```
matplotlib>=3.8.0
```

---

## 7. Ejecución y pruebas

```bash
source .venv/bin/activate
pip install matplotlib

# UI web
streamlit run app.py

# Terminal
python chat.py data/sample.xlsx
```

**Peticiones de prueba:**

```
Genera una gráfica de barras con el revenue total por región
Muéstrame la evolución mensual de las ventas en 2024
Haz un pie chart con el revenue por segmento de cliente (Enterprise, SMB, Public Sector)
Crea un gráfico de barras agrupadas comparando categorías de producto por trimestre
¿Qué 5 vendedores tienen más revenue? Ponlo en un gráfico horizontal
```

---

## 8. Puntos de discusión

1. **¿Por qué `matplotlib.use("Agg")` dentro de la tool?**
   El backend `Agg` renderiza a fichero sin necesitar una pantalla/ventana gráfica. Sin esto, en entornos sin display (servidores, Streamlit) matplotlib lanzaría un error. Se establece dentro de la función para no afectar al entorno global si el usuario importa matplotlib en otro sitio.

2. **¿Por qué `exec(compile(...))` y no `PythonAstREPLTool`?**
   `PythonAstREPLTool` captura el output de stdout, pero una figura matplotlib no imprime nada a stdout. Necesitamos ejecutar el código y luego llamar a `plt.savefig()` nosotros. Usamos `compile()` para obtener mejores mensajes de error (incluye nombre de fichero y número de línea).

3. **¿Por qué el prefijo `CHART:` y no un protocolo más sofisticado?**
   Para un ejercicio de formación, la simplicidad gana. Un prefijo de string es suficiente, no requiere serialización y es legible en logs. En producción podrías usar un `TypedDict` como return type o un campo adicional en el estado del agente.

4. **¿Cómo añadirías soporte para Plotly?**
   Cambiar `plt.savefig()` por `fig.write_image(path)` (requiere `kaleido`), o devolver el JSON del figure con `fig.to_json()` y usar `st.plotly_chart(pio.from_json(...))` en Streamlit.

5. **Los archivos PNG temporales se acumulan. ¿Cómo los limpiarías?**
   Guardar las rutas en session_state y llamar a `os.unlink(path)` en `_reset_session()`. En producción usarías un directorio temporal con `tempfile.mkdtemp()` y lo eliminarías completo al cerrar la sesión.

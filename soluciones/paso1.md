# Solución Paso 1 — Agente LangGraph para análisis de Excel

Guía paso a paso con todo el código listo para copiar y pegar.
Duración estimada: 60-90 min. 

---

## Índice

1. [Arquitectura y conceptos clave](#1-arquitectura-y-conceptos-clave)
2. [Setup del entorno](#2-setup-del-entorno)
3. [state.py — El estado del grafo](#3-statepy--el-estado-del-grafo)
4. [tools.py — Las herramientas del agente](#4-toolspy--las-herramientas-del-agente)
5. [agent.py — El grafo ReAct](#5-agentpy--el-grafo-react)
6. [chat.py — El bucle de chat en terminal](#6-chatpy--el-bucle-de-chat-en-terminal)
7. [Ejecución y pruebas](#7-ejecución-y-pruebas)
8. [Preguntas de ejemplo](#8-preguntas-de-ejemplo)

---

## 1. Arquitectura y conceptos clave

### ¿Qué es un agente ReAct?

El patrón **ReAct** (Reasoning + Acting) es un bucle donde el LLM:
1. **Razona** sobre la pregunta del usuario
2. **Decide** qué herramienta usar (o si ya tiene suficiente información)
3. **Observa** el resultado de la herramienta
4. Vuelve al paso 1 hasta tener una respuesta final

```
Usuario → LLM (razona) → Tool (ejecuta) → LLM (observa) → ... → Respuesta
```

### ¿Por qué LangGraph y no LangChain agents directamente?

LangGraph expresa el agente como un **grafo de estados** con nodos y aristas. Esto da:
- Control explícito del flujo
- Checkpointing / memoria multi-turno con `MemorySaver`
- Trazabilidad clara de cada paso

### Estructura de archivos

```
chat.py       ← CLI: carga el Excel, arranca el bucle de chat
agent.py      ← LangGraph ReAct graph (create_react_agent + MemorySaver)
tools.py      ← 3 herramientas pandas expuestas al LLM
state.py      ← AgentState TypedDict
```

---

## 2. Setup del entorno

```bash
# Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar API key
cp .env.example .env
# Editar .env: OPENAI_API_KEY=sk-...

# Generar el Excel de prueba
python generate_sample.py
```

**Verificar que funciona:**

```bash
python3 -c "import langgraph; import langchain_openai; print('OK')"
```

---

## 3. state.py — El estado del grafo

El estado es el objeto que LangGraph pasa de nodo en nodo.
En un agente de chat, el estado contiene el historial de mensajes.

**Conceptos clave:**
- `TypedDict`: define el esquema del estado con tipos estáticos
- `Annotated[list, add_messages]`: el reducer `add_messages` hace que LangGraph **añada** mensajes en lugar de reemplazar toda la lista

```python
# state.py
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

> **Nota:** `create_react_agent` gestiona este estado internamente. No necesitas instanciarlo tú. Lo definimos para entender qué ocurre bajo el capó.

---

## 4. tools.py — Las herramientas del agente

Las herramientas son las "manos" del agente: le permiten interactuar con el mundo (en este caso, con el Excel).

**Conceptos clave:**
- `@tool`: decorador que convierte una función en `BaseTool`. El **docstring** es la descripción que lee el LLM para saber cuándo y cómo usar cada herramienta.
- `PythonAstREPLTool`: ejecuta código Python parseando el AST antes (más seguro que `exec()`). Permite inyectar variables en su namespace.
- `all_sheets`: cargamos **todas** las hojas una vez y las inyectamos en el REPL para que el LLM pueda hacer `pd.merge()`.

```python
# tools.py
import io
from typing import Any

import pandas as pd
from langchain.tools import tool
from langchain_experimental.tools import PythonAstREPLTool


def _capture_df_info(df: pd.DataFrame) -> str:
    """Devuelve df.info() como string (por defecto imprime a stdout)."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def build_tools(df: pd.DataFrame, excel_path: str) -> list[Any]:
    # Cargar todas las hojas una vez para el REPL
    all_sheets: dict[str, pd.DataFrame] = pd.read_excel(
        excel_path, sheet_name=None, engine=_pick_engine(excel_path)
    )

    # ------------------------------------------------------------------
    # Herramienta 1: get_dataframe_info
    # El docstring es lo que el LLM lee para decidir cuándo usarla.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Herramienta 2: list_sheets
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Herramienta 3: python_repl
    # PythonAstREPLTool acepta `locals` para inyectar variables.
    # ------------------------------------------------------------------
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

    return [get_dataframe_info, list_sheets, repl]


def _pick_engine(path: str) -> str:
    """Selecciona el engine correcto de pandas Excel según la extensión."""
    return "xlrd" if path.lower().endswith(".xls") else "openpyxl"
```

---

## 5. agent.py — El grafo ReAct

`create_react_agent` es el constructor de alto nivel de LangGraph que crea el grafo ReAct completo: nodo LLM → nodo de herramientas → nodo LLM en bucle.

**Conceptos clave:**
- `MemorySaver`: persiste el estado del grafo en memoria, indexado por `thread_id`. Cada sesión con el mismo `thread_id` comparte historial.
- `temperature=0`: resultados deterministas, cruciales para análisis de datos.
- `prompt=SYSTEM_PROMPT`: el system prompt es el lugar de mayor impacto para controlar el comportamiento del agente.

```python
# agent.py
import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """Eres un analista de datos experto en Excel y pandas.
Tu función es ayudar a los usuarios a explorar y entender sus datos mediante lenguaje natural.

REGLAS (seguir estrictamente):
1. NUNCA modificar el DataFrame original — trata `df` como solo lectura.
2. Llama siempre a `get_dataframe_info` primero si aún no conoces el schema.
3. Tras ejecutar código en python_repl, explica los resultados en lenguaje llano.
4. Si una consulta falla, prueba una aproximación diferente antes de rendirte.
5. Formatea los resultados numéricos con claridad (miles con comas, 2 decimales).
6. Para mostrar tablas usa formato markdown para mayor legibilidad.
7. Sé conciso: responde la pregunta y para. No sobre-expliques.

VARIABLES DISPONIBLES EN python_repl:
- `df`         — DataFrame principal (primera hoja / hoja activa)
- `all_sheets` — dict {nombre_hoja: DataFrame} para todo el workbook
- `pd`         — módulo pandas

Para combinar hojas usa pd.merge():
  pd.merge(all_sheets['Sales'], all_sheets['Clients'], on='client_id')

Responde siempre en el idioma del usuario."""


def build_agent(tools: list):
    """
    Construye y compila el agente LangGraph ReAct.

    Returns
    -------
    CompiledGraph
        graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "..."}})
    """
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    # MemorySaver almacena el estado en RAM.
    # Para producción, sustituye por SqliteSaver o PostgresSaver.
    memory = MemorySaver()

    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )
```

---

## 6. chat.py — El bucle de chat en terminal

El punto de entrada que une todo: carga el Excel, construye el agente y arranca el bucle de conversación.

**Conceptos clave:**
- `thread_id` fijo por sesión: todos los turnos comparten el mismo checkpoint de `MemorySaver`, dando memoria multi-turno automática.
- `graph.invoke()` ejecuta el agente síncrono; devuelve `{"messages": [...]}`.
- El último mensaje con `content` y sin `tool_calls` es siempre la respuesta final del LLM.

```python
# chat.py
import sys
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
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
        "[dim]Escribe tu pregunta en lenguaje natural. Escribe [bold]salir[/] para terminar.[/dim]"
    )
    console.print(Panel(info, title="[bold green]Agente Analista de Excel[/]", border_style="green"))


def extract_final_answer(result: dict) -> str:
    """Extrae el texto del último AIMessage de la respuesta del agente."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return msg.content
    return "(Sin respuesta generada)"


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

    # thread_id fijo → todos los turnos comparten el mismo MemorySaver checkpoint
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


if __name__ == "__main__":
    main()
```

---

## 7. Ejecución y pruebas

```bash
# Asegúrate de tener el .venv activado y el .env con la API key
source .venv/bin/activate

# Generar datos de prueba (si no lo has hecho)
python generate_sample.py

# Arrancar el agente
python chat.py data/sample.xlsx
```

---

## 8. Preguntas de ejemplo

### Consultas simples (una hoja)

```
¿Cuántas ventas hay en total?
¿Cuál es el revenue total por trimestre?
¿Qué categoría de producto genera más ingresos?
¿Cuáles son los 5 vendedores con mayor revenue?
¿Cuántos productos tienen stock_units por debajo de 50?
```

### Consultas que combinan hojas (pd.merge)

```
¿Qué región genera más revenue en total?
¿Cuál es el revenue medio por cliente según su segmento (SMB vs Enterprise)?
¿Qué clientes llevan más de 5 años con nosotros y cuánto han comprado?
¿Qué proveedor suministra los productos más vendidos?
Muéstrame un ranking de los 10 clientes por revenue total
```

### Consultas avanzadas (multi-turno, contexto)

```
[Turno 1] ¿Cuál es el producto más vendido por unidades?
[Turno 2] ¿Y ese producto cuánto stock le queda?
[Turno 3] ¿En qué región se vende más ese producto?
```

---

## Puntos de discusión para la formación

1. **¿Por qué `PythonAstREPLTool` y no `exec()`?**
   Parseo AST previene inyección de código arbitrario. Sigue siendo poderoso pero más seguro.

2. **¿Por qué `temperature=0`?**
   En análisis de datos queremos determinismo: la misma pregunta siempre debe dar el mismo resultado.

3. **¿Qué pasa si aumentamos `temperature`?**
   El agente empieza a "inventar" valores o interpretar los datos de forma más creativa.

4. **¿Por qué `MemorySaver` y no sin memoria?**
   Sin memoria, cada pregunta es independiente. Con `MemorySaver` + `thread_id` fijo, el agente recuerda las respuestas anteriores y puede hacer análisis incremental.

5. **¿Cómo escalaría esto a producción?**
   - Sustituir `MemorySaver` por `SqliteSaver` o `PostgresSaver`
   - Añadir autenticación para el `thread_id`
   - Limitar el tamaño del historial con un `trimmer` de mensajes
   - Añadir un tool de `write_excel` para exportar resultados


#PASO 2.MD
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

#PASO 3.MD
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

#PASO 4.MD
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

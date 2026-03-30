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

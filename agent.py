"""
agent.py — Definición del agente LangGraph ReAct.

Teaching points:
    - create_react_agent es la API estable y recomendada en LangGraph 2025.
      Conecta internamente: nodo LLM → nodo de herramientas → nodo LLM en
      bucle hasta que el LLM decide que no se necesitan más llamadas a tools.
    - MemorySaver almacena el estado del grafo de conversación en memoria,
      indexado por `thread_id`. Esto da historial multi-turno sin código extra.
    - temperature=0 es intencional: queremos análisis determinista y
      reproducible, no variación creativa.
    - El system prompt establece la "personalidad" del LLM y sus reglas.

Public API
----------
    build_agent(tools) -> CompiledGraph
"""

import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Teaching point: el system prompt es el lugar con mayor impacto para
# controlar el comportamiento del agente. Mantenlo conciso, usa frases
# imperativas y ordena las reglas por importancia.
# ---------------------------------------------------------------------------
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

    Parameters
    ----------
    tools : list[BaseTool]
        Herramientas devueltas por build_tools() en tools.py.

    Returns
    -------
    CompiledGraph
        Grafo LangGraph compilado listo para invocar con:
            graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "..."}})
    """
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    # MemorySaver almacena el estado en RAM (por proceso).
    # Para producción, sustituye por SqliteSaver o PostgresSaver.
    memory = MemorySaver()

    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )

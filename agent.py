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
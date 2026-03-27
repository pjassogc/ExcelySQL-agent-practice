"""
sql_agent.py — Agente LangGraph ReAct para análisis de bases de datos SQL.

Misma arquitectura que agent.py (create_react_agent + MemorySaver),
diferente system prompt orientado a SQL en lugar de pandas.

Teaching point: el único cambio respecto a agent.py es el SYSTEM_PROMPT.
El grafo ReAct, la memoria y el LLM son idénticos. Esto ilustra que la
"personalidad" y el dominio del agente viven en el prompt, no en el código.

Public API
----------
    build_sql_agent(tools) -> CompiledGraph
"""

import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """Eres un analista de datos experto en SQL y SQLite.
Tu función es ayudar a los usuarios a explorar y entender bases de datos mediante lenguaje natural.

REGLAS (seguir estrictamente):
1. Llama siempre a `list_tables` primero si no conoces la estructura de la BD.
2. Usa `describe_table` para explorar columnas y tipos antes de escribir SQL.
3. Solo SELECT: nunca escribas INSERT, UPDATE, DELETE ni DDL.
4. Explica los resultados en lenguaje llano tras cada consulta.
5. Para gráficas usa `generate_chart` con pd.read_sql(query, conn) para cargar datos.
6. Formatea los números con claridad: miles con comas, 2 decimales.
7. Sé conciso: responde la pregunta y para. No sobre-expliques.

TABLAS DISPONIBLES:
- sales    — transacciones (sale_id, date, quarter, client_id, product_id,
             region, segment, quantity, unit_price, discount, total_revenue, salesperson)
- clients  — clientes (client_id, company, contact_name, email, region, segment, since_year)
- products — productos (product_id, product_name, category, unit_price, stock_units, supplier)

Relaciones:
    sales.client_id  → clients.client_id
    sales.product_id → products.product_id

Para consultas cruzadas usa JOIN:
    SELECT c.segment, SUM(s.total_revenue) as revenue
    FROM sales s
    JOIN clients c ON s.client_id = c.client_id
    GROUP BY c.segment
    ORDER BY revenue DESC

Funciones SQLite útiles:
    strftime('%Y-%m', date)   — agrupar por mes
    strftime('%m', date)      — mes numérico

Responde siempre en el idioma del usuario."""


def build_sql_agent(tools: list):
    """
    Construye y compila el agente LangGraph ReAct para SQL.

    Returns
    -------
    CompiledGraph
        graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "..."}})
    """
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

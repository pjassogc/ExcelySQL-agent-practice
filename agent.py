"""
agent.py — LangGraph ReAct agent definition.

Teaching points:
    - create_react_agent is the stable, recommended API in LangGraph 2025.
      It wires together: LLM node → tool node → LLM node in a loop until
      the LLM decides no more tool calls are needed.
    - MemorySaver stores the conversation graph state in memory, keyed by
      `thread_id`. This gives us multi-turn history with zero extra code.
    - temperature=0 is intentional: we want deterministic, reproducible
      analysis, not creative variation.
    - The system prompt sets the LLM's "personality" and safety rules.

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
# Teaching point: the system prompt is the single most impactful place to
# control agent behaviour. Keep it concise, use imperative sentences, and
# order rules by importance.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert data analyst assistant specializing in Excel and pandas.
Your role is to help users explore and understand their data through natural language.

RULES (follow strictly):
1. NEVER modify the original DataFrame — treat `df` as read-only.
2. Always call `get_dataframe_info` first if you don't know the schema yet.
3. After running code in python_repl, explain the results in plain language.
4. If a query fails, try a different approach before giving up.
5. Format numeric results clearly (e.g., use commas for thousands, 2 decimal places).
6. When showing tables, use markdown format for readability.
7. Be concise: answer the question, then stop. Do not over-explain.

AVAILABLE VARIABLES IN python_repl:
- `df`         — primary DataFrame (first/active sheet)
- `all_sheets` — dict {sheet_name: DataFrame} for the full workbook
- `pd`         — pandas module

If the user asks about a topic unrelated to data analysis, politely redirect them."""


def build_agent(tools: list):
    """
    Build and compile the LangGraph ReAct agent.

    Parameters
    ----------
    tools : list[BaseTool]
        Tools returned by build_tools() in tools.py.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to invoke with:
            graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "..."}})
    """
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    # MemorySaver stores state in RAM (per process).
    # For production, swap with SqliteSaver or PostgresSaver.
    memory = MemorySaver()

    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )

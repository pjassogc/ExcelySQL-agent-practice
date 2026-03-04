"""
state.py — Typed agent state definition.

Teaching point: TypedDict + Annotated is the idiomatic way to define LangGraph state.
The `add_messages` reducer merges new messages into the existing list (append semantics),
which is what enables multi-turn conversation history.

LangGraph's create_react_agent manages this state internally, but we expose it here
so learners understand what's happening under the hood.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State schema for the Excel analyst ReAct agent.

    Fields
    ------
    messages : list
        Conversation history. The `add_messages` annotation tells LangGraph to
        *append* new messages rather than replace the list on each graph update.
        This is the core mechanism behind multi-turn memory.
    """

    messages: Annotated[list, add_messages]

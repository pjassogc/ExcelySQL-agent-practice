"""
state.py — Typed agent state definition.

Teaching point: TypedDict + Annotated es la forma idiomática de definir
estado en LangGraph. El reducer `add_messages` fusiona los nuevos mensajes
en la lista existente (semántica de append), lo que habilita el historial
multi-turno.

create_react_agent gestiona este estado internamente, pero lo exponemos aquí
para que los alumnos entiendan qué ocurre bajo el capó.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Esquema de estado para el agente ReAct de análisis de Excel.

    Fields
    ------
    messages : list
        Historial de conversación. La anotación `add_messages` indica a
        LangGraph que *añada* los nuevos mensajes en lugar de reemplazar la
        lista completa en cada actualización del grafo.
        Este es el mecanismo central de la memoria multi-turno.
    """

    messages: Annotated[list, add_messages]

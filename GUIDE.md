# Guión del Formador — Agente LangGraph para Excel

> **Audiencia**: Desarrolladores Python senior
> **Duración total**: ~70 minutos
> **Formato**: Sesión presencial o remota con pantalla compartida
> **Requisitos previos**: Python intermedio/avanzado, familiaridad con APIs REST

---

## Antes de empezar (preparación)

- [ ] Terminal abierta en la carpeta del proyecto
- [ ] `.env` configurado con `OPENAI_API_KEY` válida
- [ ] `pip install -r requirements.txt` completado
- [ ] `data/sample.xlsx` disponible (ejecutar `python generate_sample.py` si no existe)
- [ ] Editor de código abierto en `tools.py` y `agent.py`

---

## Sección 1 — Introducción (5 min)

### Qué decir

> "Vamos a construir un agente de inteligencia artificial que permite cargar cualquier archivo Excel y hacerle preguntas en lenguaje natural desde la terminal. No necesitas recordar sintaxis de pandas — el agente escribe el código por ti."

**Mostrar en pantalla**: La demo funcionando (2-3 preguntas rápidas sobre `sample.xlsx`).

### Por qué es relevante

- Los analistas de datos pasan horas escribiendo código exploratorio repetitivo
- Un agente reduce ese tiempo drásticamente para tareas de exploración
- Este patrón (LLM + herramientas + memoria) es la base de casi todos los sistemas de IA productivos en 2025

### Mensaje clave para los asistentes

> "El agente no reemplaza a un ingeniero de datos. Lo amplifica. El LLM razona, pero el desarrollador controla qué herramientas existen, qué puede hacer el agente y cuáles son sus límites."

---

## Sección 2 — Conceptos clave (10 min)

### 2.1 ¿Qué es un agente LLM?

Usar esta analogía:

> "Imagina un jefe de proyecto muy inteligente que no sabe usar Excel directamente, pero tiene un equipo de especialistas (herramientas). El jefe recibe tu pregunta, decide qué especialista llamar, recibe el resultado y te da una respuesta. Puede hacer varias consultas a distintos especialistas antes de responderte."

Conceptos que cubrir:

| Concepto | Explicación simple |
|---|---|
| **LLM** | El "razonador": entiende la pregunta y decide qué hacer |
| **Tool / Herramienta** | Una función Python que el LLM puede invocar |
| **Tool Call** | El LLM emite un JSON con `{tool: "...", args: {...}}` |
| **Observation** | El resultado de ejecutar la herramienta |
| **ReAct loop** | Reason → Act → Observe → Reason → … hasta respuesta final |

### 2.2 ¿Qué es LangGraph?

> "LangGraph modela el flujo del agente como un **grafo** (nodos + aristas). En el caso más simple — que es el nuestro — hay dos nodos: el nodo LLM y el nodo de herramientas. El LLM decide si pasar al nodo de herramientas o terminar."

**Mostrar el diagrama Mermaid del README.**

### 2.3 Por qué LangGraph y no `create_pandas_dataframe_agent`

> "LangChain tiene una función llamada `create_pandas_dataframe_agent` en el paquete `langchain_experimental`. Funciona, pero es una caja negra. LangGraph nos da visibilidad total del grafo, control del estado y soporte nativo para memoria multi-turno. En 2025 es la API recomendada."

---

## Sección 3 — Arquitectura del proyecto (10 min)

### Recorrer fichero a fichero (resumen)

Abrir cada fichero y señalar **una sola cosa clave** por fichero:

| Fichero | Punto clave |
|---|---|
| `state.py` | `Annotated[list, add_messages]` — el reducer que habilita la memoria |
| `tools.py` | `@tool` + docstring = lo que el LLM "lee" para decidir qué herramienta usar |
| `agent.py` | `MemorySaver` + `thread_id` = multi-turno automático |
| `chat.py` | `graph.invoke(messages, config)` — API de invocación del grafo |

### Pregunta para activar la clase

> "¿Qué pasaría si no tuviéramos `MemorySaver`? ¿El agente recordaría lo que dijiste en el turno anterior?"

Respuesta esperada: No. Cada invocación sería independiente. Con `MemorySaver` + `thread_id`, LangGraph almacena el estado completo del grafo y lo recupera en el siguiente turno.

---

## Sección 4 — Walkthrough del código (20 min)

### 4.1 `state.py` (2 min)

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

> "Este es todo el estado del agente. `add_messages` es un **reducer**: en lugar de reemplazar la lista de mensajes, los **acumula**. Es el mecanismo de memoria más simple posible."

### 4.2 `tools.py` (8 min)

**Punto 1: el decorador `@tool`**

```python
@tool
def get_dataframe_info(dummy: str = "") -> str:
    """
    Return schema and sample data for the active DataFrame.
    Use this tool FIRST when the user asks a question about the data...
    """
```

> "El decorador convierte esta función en un objeto `BaseTool`. El LLM recibe el nombre y el **docstring** de cada herramienta. Por eso los docstrings aquí no son documentación para humanos — son instrucciones para el LLM. Escribirlos bien es crítico."

**Punto 2: `PythonAstREPLTool`**

```python
repl = PythonAstREPLTool(
    locals={"df": df, "all_sheets": all_sheets, "pd": pd},
    ...
)
```

> "El REPL ejecuta código Python via AST parsing. Esto es más seguro que `exec()` porque el parser rechaza ciertas construcciones peligrosas. Además, inyectamos `df` y `all_sheets` en el namespace del REPL, así el LLM puede escribir `df.groupby(...)` directamente sin saber cómo cargar el archivo."

**Punto 3: closures de Python**

> "Fíjense que `get_dataframe_info` y `list_sheets` están definidas dentro de `build_tools`. Esto es un closure: capturan `df`, `all_sheets` y `excel_path` del scope externo. Cada vez que llamamos a `build_tools` con un archivo diferente, obtenemos herramientas apuntando a ese archivo específico."

### 4.3 `agent.py` (5 min)

```python
return create_react_agent(
    llm,
    tools=tools,
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)
```

> "Esta única llamada construye el grafo completo: nodo LLM, nodo de herramientas, aristas condicionales, y el checkpointer de memoria. En LangGraph podríamos construirlo manualmente nodo a nodo, pero `create_react_agent` es la abstracción recomendada para el patrón ReAct estándar."

**Sobre el system prompt:**

> "Las reglas más importantes son las dos primeras: nunca modificar `df`, y siempre llamar a `get_dataframe_info` si no conocemos el schema. Sin estas restricciones, el LLM podría escribir código que corrompe el DataFrame o que falla porque no sabe qué columnas existen."

### 4.4 `chat.py` (5 min)

```python
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

result = graph.invoke(
    {"messages": [("human", user_input)]},
    config=config,
)
```

> "El `thread_id` es la clave de memoria. LangGraph busca el checkpoint guardado bajo ese ID y restaura el estado completo antes de ejecutar el grafo. Al usar un ID fijo por sesión, cada pregunta del chat accede al historial completo."

---

## Sección 5 — Demo en vivo (10 min)

### Secuencia de preguntas recomendada

```
1. "What columns does this file have?"
   → el agente llama a get_dataframe_info

2. "What is the total sales by region?"
   → llama a python_repl con df.groupby(...)

3. "Now show me only the top 3"
   → reutiliza el contexto anterior (prueba de multi-turno)

4. "What sheets does the workbook have?"
   → llama a list_sheets

5. "Show me the first 5 rows of the Clients sheet"
   → python_repl con all_sheets['Clients'].head()
```

### Qué señalar durante la demo

- El panel "Agent thinking…" aparece mientras el agente itera
- El agente puede hacer múltiples tool calls antes de responder
- Las referencias a respuestas anteriores ("the top 3", "those products") funcionan

---

## Sección 6 — Ejercicios prácticos (15 min)

Ver `exercises/exercise_1.md`, `exercise_2.md`, `exercise_3.md`.

### Dinámica recomendada

- **Ejercicio 1** (5 min): individual, cambio de una línea
- **Ejercicio 2** (10 min): parejas, añadir una herramienta nueva
- **Ejercicio 3**: opcional para quienes terminen antes

### Pistas si se atascan

**Ejercicio 2**: recordar que la herramienta va dentro de `build_tools()` y debe añadirse a la lista de retorno.

**Ejercicio 3**: `MemorySaver` no persiste entre procesos. Para persistencia real, necesitan `SqliteSaver`.

---

## Cierre (5 min)

### Qué llevarse

1. **El patrón es universal**: LangGraph + tools funciona para cualquier fuente de datos, no solo Excel
2. **Los docstrings son prompts**: la calidad de la descripción de la herramienta determina cuándo el LLM la usa
3. **MemorySaver + thread_id = multi-turno gratis**: no hay que gestionar historial manualmente

### Próximos pasos sugeridos

- Añadir persistencia real con `SqliteSaver`
- Añadir streaming de respuestas con `graph.stream()`
- Conectar a una base de datos en lugar de Excel (misma arquitectura, diferente herramienta)
- Añadir guardrails (filtrar preguntas fuera de scope con un router antes del agente)

---

## FAQ — Preguntas frecuentes de asistentes

**¿Es seguro dar acceso al LLM a ejecutar código Python arbitrario?**

> `PythonAstREPLTool` añade una capa de protección, pero en producción se recomienda ejecutar el REPL en un sandbox (contenedor Docker con permisos limitados, o una solución como `e2b`). Para uso interno con datos no sensibles, el riesgo es bajo.

**¿Funciona con archivos Excel muy grandes (millones de filas)?**

> El cuello de botella es la memoria RAM al cargar el DataFrame. Para archivos grandes, se recomienda cargar solo las columnas necesarias con `usecols`, o muestrear el DataFrame antes de pasarlo al agente.

**¿Se puede usar con un modelo local (Ollama, LM Studio)?**

> Sí. Sustituir `ChatOpenAI` por `ChatOllama` de `langchain-ollama`. El modelo local debe soportar tool calling (function calling). `llama3.1`, `mistral-nemo` y `qwen2.5` lo soportan.

**¿Por qué no usar LangChain LCEL en lugar de LangGraph?**

> LCEL (chains) es ideal para flujos lineales y predecibles. LangGraph añade loops, condicionales y estado persistente. Para agentes que necesitan iterar hasta encontrar una respuesta, LangGraph es la herramienta correcta.

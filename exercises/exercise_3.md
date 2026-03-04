# Ejercicio 3 — Soporte multi-hoja con `switch_sheet`

**Dificultad**: Avanzado (20 minutos)
**Objetivo**: Entender cómo el estado del agente puede cambiar dinámicamente durante una sesión.

---

## Enunciado

Actualmente el agente carga la primera hoja del Excel como `df` y mantiene ese DataFrame durante toda la sesión.
Tu tarea es añadir una herramienta `switch_sheet` que permita al usuario cambiar la hoja activa.

**Desafío**: El REPL y las herramientas necesitan acceder al DataFrame actualizado.
Esto requiere pensar en cómo compartir estado mutable entre herramientas.

---

## Problema de diseño

```
# El problema: df es una variable local en build_tools()
# Si el usuario dice "switch to the Clients sheet", ¿cómo actualiza
# switch_sheet el df que usa python_repl?
```

Hay varias soluciones. La más sencilla usa un **objeto contenedor mutable**:

```python
class DataFrameHolder:
    """Mutable container that lets tools share a reference to the active df."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
```

Dado que las herramientas capturan el **objeto** (no el valor), actualizar `holder.df` afecta a todas las herramientas que tengan referencia a `holder`.

---

## Pasos

### Paso 1 — Crear `DataFrameHolder`

En `tools.py`, añade la clase `DataFrameHolder` antes de `build_tools()`.

### Paso 2 — Refactorizar `build_tools()`

```python
def build_tools(df: pd.DataFrame, excel_path: str) -> list:
    all_sheets = pd.read_excel(excel_path, sheet_name=None, engine=_pick_engine(excel_path))
    holder = DataFrameHolder(df)

    @tool
    def get_dataframe_info(dummy: str = "") -> str:
        """... (usa holder.df en lugar de df) ..."""
        info = _capture_df_info(holder.df)
        # ...

    @tool
    def switch_sheet(sheet_name: str) -> str:
        """
        Switch the active DataFrame to a different sheet in the workbook.

        Use this tool when the user asks to analyze a different sheet,
        or mentions a sheet by name.

        Parameters
        ----------
        sheet_name : str
            Exact name of the sheet to switch to (case-sensitive).
        """
        # Tu implementación aquí
        ...

    repl = PythonAstREPLTool(
        locals={"df": holder, "all_sheets": all_sheets, "pd": pd},
        # Nota: el LLM tendrá que usar holder.df en lugar de df
        # O bien actualizar el locals dict — ¿cuál es mejor? ¿por qué?
        ...
    )
```

### Paso 3 — Actualizar el system prompt

Añade una nota en `SYSTEM_PROMPT` en `agent.py` indicando que `switch_sheet` está disponible y cuándo usarla.

### Paso 4 — Probar

```
You: What sheets does this workbook have?
You: Switch to the Clients sheet
You: What are the top 5 clients by total purchases?
You: Go back to the Sales sheet and compare with those clients
```

---

## Preguntas de reflexión

1. ¿Por qué usar un objeto mutable en lugar de actualizar el dict `locals` del REPL?
2. ¿Qué problemas de concurrencia podría tener este enfoque en un servidor web?
3. ¿Cómo guardarías la hoja activa en el estado de LangGraph en lugar de en memoria local?

---

## Pistas

- `all_sheets.get(sheet_name)` devuelve `None` si la hoja no existe — maneja ese caso
- Considera hacer la comparación de nombres case-insensitive para mejor UX
- El REPL no se puede re-crear fácilmente una vez inicializado; el truco del `holder` evita ese problema

---

> Ver solución completa en `solutions/solution_3.py`

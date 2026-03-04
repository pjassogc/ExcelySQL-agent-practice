# Ejercicio 2 — Añadir la herramienta `export_summary`

**Dificultad**: Media (10 minutos)
**Objetivo**: Aprender a crear y registrar una herramienta nueva en el agente.

---

## Enunciado

El agente actualmente puede analizar datos, pero no puede guardar resultados.
Tu tarea es añadir una herramienta `export_summary` que:

1. Acepta un texto de resumen como parámetro
2. Lo guarda en un archivo `summary.txt` en el directorio actual
3. Devuelve confirmación con la ruta del archivo guardado

---

## Pasos

### Paso 1 — Añadir la herramienta en `tools.py`

Dentro de la función `build_tools()`, justo antes del `return`, añade:

```python
@tool
def export_summary(text: str) -> str:
    """
    Save a text summary to a file called summary.txt.

    Use this tool when the user asks to save, export, or write down
    the results of the analysis. Pass the full summary text as the argument.

    Parameters
    ----------
    text : str
        The summary text to save.
    """
    # Tu implementación aquí
    ...
```

### Paso 2 — Registrar la herramienta

Modifica el `return` de `build_tools()` para incluir la nueva herramienta:

```python
return [get_dataframe_info, list_sheets, repl, export_summary]
```

### Paso 3 — Probar

Reinicia el agente y prueba:

```
You: Analyze total sales by region and save a summary to a file
```

---

## Criterios de éxito

- [ ] El archivo `summary.txt` se crea en el directorio actual
- [ ] El contenido del archivo refleja el resumen generado por el agente
- [ ] El agente confirma que el archivo se ha guardado con la ruta correcta

---

## Pistas

- Usa `pathlib.Path` para construir rutas de forma robusta
- El parámetro `text` contiene el texto que el LLM ha decidido guardar
- Recuerda añadir `from pathlib import Path` si no está ya importado

---

> Ver solución completa en `solutions/solution_2.py`

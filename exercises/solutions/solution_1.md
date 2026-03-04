# Solución Ejercicio 1 — Cambiar el modelo LLM

## Cambio mínimo: editar `.env`

```env
# .env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

El agente lee esta variable en `agent.py`:
```python
model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
```

No es necesario modificar ningún archivo Python.

---

## Extensión: seleccionar modelo desde CLI

Si quisieras pasar el modelo como argumento `--model`, modificarías `chat.py` y `agent.py`:

### `chat.py` — Añadir argumento CLI

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Excel Analyst Agent")
    parser.add_argument("excel_file", nargs="?", help="Path to Excel file")
    parser.add_argument("--model", default=None, help="OpenAI model (e.g. gpt-4o)")
    return parser.parse_args()

def main():
    args = parse_args()
    # ...
    graph = build_agent(tools, model=args.model)
```

### `agent.py` — Aceptar modelo como parámetro

```python
def build_agent(tools: list, model: str | None = None):
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)
    # ...
```

### Uso

```bash
python chat.py data/sample.xlsx --model gpt-4o
```

---

## Reflexión: ¿cuándo usar gpt-4o?

| Situación | Recomendación |
|---|---|
| Exploración rápida, preguntas simples | `gpt-4o-mini` (más barato, suficientemente bueno) |
| Análisis complejos, código pandas avanzado | `gpt-4o` (mejor razonamiento) |
| Producción con muchos usuarios | `gpt-4o-mini` primero, escalar si es necesario |
| Prototipado y demos | `gpt-4o-mini` para no gastar créditos |

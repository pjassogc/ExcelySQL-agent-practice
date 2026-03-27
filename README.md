# Ejercicio: Agente LangGraph para análisis de Excel

Construye un agente conversacional en Python que permita hacer preguntas en lenguaje natural sobre un archivo Excel con datos de ventas.

## Objetivo

El agente debe ser capaz de:

- Explorar la estructura del Excel (hojas, columnas, tipos de datos)
- Responder preguntas simples: _"¿Cuántas ventas hay en Q3?"_
- Combinar hojas para consultas cruzadas: _"¿Qué región genera más revenue por cliente de segmento Enterprise?"_
- Mantener contexto entre preguntas (memoria multi-turno)

## Arquitectura objetivo

```
chat.py          ← CLI: carga el Excel, arranca el bucle de chat
  └── agent.py   ← LangGraph ReAct graph (create_react_agent + MemorySaver)
       └── tools.py  ← 3 herramientas pandas expuestas al LLM
            state.py ← AgentState TypedDict
```

## Datos disponibles

El archivo `data/sample.xlsx` tiene 3 hojas:

| Hoja      | Filas | Columnas clave |
|-----------|-------|----------------|
| Sales     | 200   | sale_id, date, quarter, client_id, product_id, region, segment, total_revenue |
| Clients   | 40    | client_id, company, region, segment, since_year |
| Products  | 20    | product_id, product_name, category, unit_price, stock_units |

Las hojas están relacionadas por `client_id` y `product_id`, lo que permite hacer `pd.merge()` para consultas cruzadas.

Genera el archivo con:

```bash
python generate_sample.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edita .env y añade tu OPENAI_API_KEY
```

## Herramientas que debe implementar el agente

| Tool | Qué hace |
|------|----------|
| `get_dataframe_info` | Schema + estadísticas + head(5) de la hoja activa |
| `list_sheets` | Lista las hojas disponibles con número de filas |
| `python_repl` | Ejecuta código pandas arbitrario con `df`, `all_sheets` y `pd` inyectados |

## Cómo ejecutar (cuando esté implementado)

```bash
python chat.py data/sample.xlsx
```

## Solución de referencia

La solución completa está en la rama `soluciones/paso1`:

```bash
git checkout soluciones/paso1
```

La guía paso a paso con todo el código está en `soluciones/paso1.md`.

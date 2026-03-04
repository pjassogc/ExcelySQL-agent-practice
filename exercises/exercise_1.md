# Ejercicio 1 — Cambiar el modelo LLM

**Dificultad**: Fácil (5 minutos)
**Objetivo**: Entender cómo configurar el modelo y observar diferencias de comportamiento.

---

## Enunciado

El agente usa `gpt-4o-mini` por defecto. Tu tarea es:

1. Cambiar el modelo a `gpt-4o` modificando el archivo `.env`
2. Reiniciar el agente y hacer las **mismas preguntas** que hiciste antes
3. Observar las diferencias en calidad, velocidad y coste

---

## Pasos

### Paso 1 — Editar `.env`

Abre el archivo `.env` y cambia:

```env
OPENAI_MODEL=gpt-4o-mini
```

por:

```env
OPENAI_MODEL=gpt-4o
```

### Paso 2 — Reiniciar el agente

```bash
python chat.py data/sample.xlsx
```

### Paso 3 — Hacer estas preguntas y comparar respuestas

```
What is the total revenue by region and product category combined?
Are there any interesting patterns or anomalies in the data?
```

---

## Preguntas de reflexión

1. ¿`gpt-4o` produce respuestas notablemente mejores para análisis de datos?
2. ¿Cuándo valdría la pena pagar el coste extra de `gpt-4o`?
3. ¿Cómo cambiarías el código para que el usuario pudiera elegir el modelo desde la línea de comandos?

---

## Pista

El modelo se configura en `agent.py`:

```python
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)
```

`os.getenv("OPENAI_MODEL", "gpt-4o-mini")` lee la variable de entorno, con `gpt-4o-mini` como fallback.
Para añadir soporte por CLI, podrías pasar el nombre del modelo como argumento a `build_agent()`.

---

> Ver solución en `solutions/solution_1.md`

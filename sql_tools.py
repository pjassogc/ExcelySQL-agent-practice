# sql_tools.py
import sqlite3
import tempfile
from typing import Any

import pandas as pd
from langchain.tools import tool


def build_sql_tools(db_path: str) -> list[Any]:
    # check_same_thread=False: Streamlit puede invocar tools desde hilos distintos
    conn = sqlite3.connect(db_path, check_same_thread=False)

    @tool
    def list_tables(dummy: str = "") -> str:
        """
        Lista todas las tablas disponibles en la base de datos SQLite.

        Usa esta herramienta PRIMERO para explorar la estructura de la BD
        antes de escribir consultas SQL.
        """
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        if not tables:
            return "No hay tablas en la base de datos."
        return "Tablas disponibles:\n" + "\n".join(f"  - {t}" for t in tables)

    @tool
    def describe_table(table_name: str) -> str:
        """
        Describe el schema de una tabla: columnas, tipos, nulos y filas de muestra.

        Args:
            table_name: Nombre de la tabla (sales, clients, products)
        """
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        cols = cursor.fetchall()
        if not cols:
            available = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            return (
                f"Tabla '{table_name}' no encontrada. "
                f"Disponibles: {[r[0] for r in available]}"
            )

        col_lines = [
            f"  {c[1]:20s} {c[2]:15s}" + (" NOT NULL" if c[3] else "")
            for c in cols
        ]
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        sample = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 3", conn)

        return (
            f"Tabla: {table_name} ({count:,} filas)\n"
            "Columnas:\n" + "\n".join(col_lines) +
            f"\n\nPrimeras 3 filas:\n{sample.to_markdown(index=False)}"
        )

    @tool
    def run_sql(query: str) -> str:
        """
        Ejecuta una consulta SQL SELECT y devuelve los resultados en formato tabla.

        Solo se permiten SELECT. Para consultas cruzadas usa JOIN:
            SELECT s.region, SUM(s.total_revenue) as revenue
            FROM sales s
            JOIN clients c ON s.client_id = c.client_id
            GROUP BY s.region
            ORDER BY revenue DESC

        Args:
            query: Consulta SQL SELECT válida.
        """
        if not query.strip().upper().startswith("SELECT"):
            return "Error: solo se permiten consultas SELECT."
        try:
            df = pd.read_sql(query, conn)
            if df.empty:
                return "La consulta no devolvió resultados."
            return df.to_markdown(index=False)
        except Exception as exc:
            return f"Error SQL: {exc}"

    @tool
    def generate_chart(code: str) -> str:
        """
        Genera una gráfica ejecutando código Python y la guarda como PNG.

        IMPORTANTE: NO llames a plt.show(). La figura se guarda automáticamente.

        Args:
            code: Código Python que crea una figura matplotlib.

        Variables disponibles:
            conn — conexión sqlite3 (usa pd.read_sql(query, conn) para datos)
            pd   — pandas
            plt  — matplotlib.pyplot

        Ejemplos:

            # Revenue por región
            df = pd.read_sql(
                "SELECT region, SUM(total_revenue) as revenue FROM sales GROUP BY region ORDER BY revenue",
                conn
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            df.plot(kind='barh', x='region', y='revenue', ax=ax, legend=False)
            ax.set_title('Revenue por región')

            # Ventas mensuales
            df = pd.read_sql(
                "SELECT strftime('%Y-%m', date) as mes, SUM(total_revenue) as revenue FROM sales GROUP BY mes ORDER BY mes",
                conn
            )
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['mes'], df['revenue'], marker='o')
            ax.set_title('Revenue mensual 2024')
            plt.xticks(rotation=45)

            # JOIN entre tablas
            df = pd.read_sql(
                "SELECT c.segment, SUM(s.total_revenue) as revenue FROM sales s JOIN clients c ON s.client_id = c.client_id GROUP BY c.segment",
                conn
            )
            fig, ax = plt.subplots()
            df.plot(kind='pie', y='revenue', labels=df['segment'], autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            ax.set_title('Revenue por segmento')

        Returns:
            "CHART:/ruta/al/archivo.png" si tiene éxito, o mensaje de error.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        local_ns = {"conn": conn, "pd": pd, "plt": plt}

        try:
            exec(compile(code, "<chart>", "exec"), local_ns)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="chart_")
            plt.savefig(tmp.name, bbox_inches="tight", dpi=150)
            plt.close("all")
            return f"CHART:{tmp.name}"
        except Exception as exc:
            plt.close("all")
            return f"Error al generar la gráfica: {exc}"

    return [list_tables, describe_table, run_sql, generate_chart]
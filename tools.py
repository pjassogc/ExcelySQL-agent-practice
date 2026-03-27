"""
tools.py — Herramientas LangChain para el agente analista de Excel.

Se exponen cuatro herramientas:
    1. get_dataframe_info  — Schema + muestra de datos (llama primero a esta)
    2. list_sheets         — Lista todas las hojas del workbook
    3. python_repl         — Ejecuta código pandas arbitrario vía parsing AST
    4. generate_chart      — Genera gráficas matplotlib y devuelve la ruta PNG

Teaching points:
    - El decorador @tool convierte una función normal en un BaseTool de LangChain.
      El docstring se convierte en la descripción que lee el LLM para decidir
      cuándo usar cada herramienta.
    - PythonAstREPLTool es más seguro que exec() porque parsea vía AST antes
      de ejecutar.
    - Inyectamos `df` y `all_sheets` en el namespace del REPL para que el LLM
      pueda referenciarlos directamente sin necesidad de cargar el archivo.

Public API
----------
    build_tools(df, excel_path) -> list[BaseTool]
"""

import io
import tempfile
from typing import Any

import pandas as pd
from langchain.tools import tool
from langchain_experimental.tools import PythonAstREPLTool


def _capture_df_info(df: pd.DataFrame) -> str:
    """Devuelve df.info() como string (por defecto imprime a stdout)."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def build_tools(df: pd.DataFrame, excel_path: str) -> list[Any]:
    """
    Construye y devuelve la lista de herramientas disponibles para el agente.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame principal (primera hoja o la elegida por el usuario).
    excel_path : str
        Ruta absoluta o relativa al archivo Excel, usada por list_sheets.

    Returns
    -------
    list[BaseTool]
        Herramientas listas para pasar a create_react_agent.
    """
    # Cargamos todas las hojas una vez para que el REPL pueda acceder a cualquiera.
    all_sheets: dict[str, pd.DataFrame] = pd.read_excel(
        excel_path, sheet_name=None, engine=_pick_engine(excel_path)
    )

    # ------------------------------------------------------------------
    # Herramienta 1: get_dataframe_info
    # ------------------------------------------------------------------
    @tool
    def get_dataframe_info(dummy: str = "") -> str:
        """
        Devuelve el schema y datos de muestra del DataFrame activo (`df`).

        Usa esta herramienta PRIMERO cuando el usuario haga una pregunta sobre
        los datos, antes de escribir código pandas. Proporciona:
          - Nombres de columna y tipos de dato (df.info())
          - Estadísticas descriptivas (df.describe())
          - Primeras 5 filas (df.head())

        El parámetro `dummy` no se usa; pasa una cadena vacía o cualquier valor.
        """
        info = _capture_df_info(df)
        describe = df.describe(include="all").to_string()
        head = df.head(5).to_markdown(index=False)
        return (
            f"=== df.info() ===\n{info}\n"
            f"=== df.describe() ===\n{describe}\n\n"
            f"=== df.head(5) ===\n{head}"
        )

    # ------------------------------------------------------------------
    # Herramienta 2: list_sheets
    # ------------------------------------------------------------------
    @tool
    def list_sheets(dummy: str = "") -> str:
        """
        Lista todos los nombres de hojas disponibles en el workbook Excel.

        Usa esta herramienta cuando el usuario pregunte sobre la estructura del
        archivo, quiera saber cuántas hojas existen, o mencione una hoja por nombre.

        En el python_repl, todas las hojas son accesibles via el dict `all_sheets`:
            all_sheets['NombreHoja']  ->  pd.DataFrame

        Para combinar hojas usa pd.merge():
            pd.merge(all_sheets['Sales'], all_sheets['Clients'], on='client_id')

        El parámetro `dummy` no se usa; pasa una cadena vacía o cualquier valor.
        """
        names = list(all_sheets.keys())
        lines = [f"  {i + 1}. '{name}' — {len(all_sheets[name])} filas" for i, name in enumerate(names)]
        return "Hojas disponibles en el workbook:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Herramienta 3: python_repl
    # ------------------------------------------------------------------
    # Inyectamos df y all_sheets en el namespace del REPL.
    repl = PythonAstREPLTool(
        locals={"df": df, "all_sheets": all_sheets, "pd": pd},
        description=(
            "Ejecuta código Python / pandas para analizar los datos del Excel.\n"
            "Variables disponibles:\n"
            "  df          — DataFrame principal (solo lectura, NO reasignar)\n"
            "  all_sheets  — dict que mapea nombre de hoja -> DataFrame\n"
            "  pd          — módulo pandas\n\n"
            "Reglas:\n"
            "  - NUNCA modificar df en su lugar (no df.drop(inplace=True), etc.)\n"
            "  - Usa .to_markdown() para output de tablas legible\n"
            "  - Mantén el código conciso; una expresión por paso de análisis\n"
            "  - Si una operación falla, captura el error y prueba otra aproximación\n\n"
            "Ejemplos:\n"
            "  df.groupby('category')['total_revenue'].sum().sort_values(ascending=False)\n"
            "  pd.merge(all_sheets['Sales'], all_sheets['Clients'], on='client_id').groupby('segment')['total_revenue'].sum()\n"
            "  all_sheets['Products'].sort_values('unit_price', ascending=False).head(5)"
        ),
    )

    # ------------------------------------------------------------------
    # Herramienta 4: generate_chart
    # ------------------------------------------------------------------
    # Usamos exec() en lugar de PythonAstREPLTool porque necesitamos
    # ejecutar el código Y luego llamar a plt.savefig() nosotros.
    # compile() proporciona mejores mensajes de error que exec() directo.
    # matplotlib.use("Agg") establece el backend sin ventana gráfica,
    # necesario en servidores y en Streamlit.
    # ------------------------------------------------------------------
    @tool
    def generate_chart(code: str) -> str:
        """
        Genera una gráfica ejecutando código matplotlib y la guarda como PNG.

        IMPORTANTE: NO llames a plt.show(). La figura se guarda automáticamente.

        Args:
            code: Código Python que crea una figura matplotlib.

        Variables disponibles:
            df          — DataFrame principal
            all_sheets  — dict {nombre_hoja: DataFrame}
            pd          — pandas
            plt         — matplotlib.pyplot

        Ejemplos:

            # Gráfico de barras horizontal por región
            fig, ax = plt.subplots(figsize=(10, 6))
            df.groupby('region')['total_revenue'].sum().sort_values().plot(kind='barh', ax=ax)
            ax.set_title('Revenue por región')

            # Serie temporal mensual
            fig, ax = plt.subplots(figsize=(12, 5))
            df.set_index('date')['total_revenue'].resample('ME').sum().plot(ax=ax)
            ax.set_title('Revenue mensual 2024')

            # Pie chart combinando hojas
            fig, ax = plt.subplots(figsize=(8, 8))
            merged = pd.merge(all_sheets['Sales'], all_sheets['Clients'], on='client_id')
            merged.groupby('segment')['total_revenue'].sum().plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title('Revenue por segmento de cliente')

        Returns:
            "CHART:/ruta/al/archivo.png" si tiene éxito, o mensaje de error.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        local_ns = {
            "df": df,
            "all_sheets": all_sheets,
            "pd": pd,
            "plt": plt,
        }

        try:
            exec(compile(code, "<chart>", "exec"), local_ns)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="chart_")
            plt.savefig(tmp.name, bbox_inches="tight", dpi=150)
            plt.close("all")
            return f"CHART:{tmp.name}"
        except Exception as exc:
            plt.close("all")
            return f"Error al generar la gráfica: {exc}"

    return [get_dataframe_info, list_sheets, repl, generate_chart]


def _pick_engine(path: str) -> str:
    """Selecciona el engine correcto de pandas Excel según la extensión."""
    return "xlrd" if path.lower().endswith(".xls") else "openpyxl"

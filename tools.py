"""
tools.py — Custom LangChain tools for the Excel analyst agent.

Three tools are exposed:
    1. get_dataframe_info  — Schema + sample data (always call this first)
    2. list_sheets         — List all sheets in the workbook
    3. python_repl         — Execute arbitrary pandas code via AST parsing

Teaching points:
    - @tool decorator converts a plain function into a LangChain BaseTool.
    - The docstring becomes the tool description the LLM reads to decide when to use it.
    - PythonAstREPLTool is safer than exec() because it parses via AST before running.
    - We inject `df` and `all_sheets` into the REPL namespace so the LLM can reference
      them directly without needing to know how to load the file.

Public API
----------
    build_tools(df, excel_path) -> list[BaseTool]
"""

import io
from typing import Any

import pandas as pd
from langchain.tools import tool
from langchain_experimental.tools import PythonAstREPLTool


def _capture_df_info(df: pd.DataFrame) -> str:
    """Return df.info() as a string (it prints to stdout by default)."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def build_tools(df: pd.DataFrame, excel_path: str) -> list[Any]:
    """
    Build and return the list of tools available to the agent.

    Parameters
    ----------
    df : pd.DataFrame
        The primary DataFrame (first sheet or sheet chosen by the user).
    excel_path : str
        Absolute or relative path to the Excel file, used by list_sheets.

    Returns
    -------
    list[BaseTool]
        Tools ready to be passed to create_react_agent.
    """
    # Load all sheets once so the REPL can access any of them.
    all_sheets: dict[str, pd.DataFrame] = pd.read_excel(
        excel_path, sheet_name=None, engine=_pick_engine(excel_path)
    )

    # ------------------------------------------------------------------
    # Tool 1: get_dataframe_info
    # ------------------------------------------------------------------
    @tool
    def get_dataframe_info(dummy: str = "") -> str:
        """
        Return schema and sample data for the active DataFrame (`df`).

        Use this tool FIRST when the user asks a question about the data,
        before writing any pandas code. It provides:
          - Column names and dtypes (df.info())
          - Descriptive statistics (df.describe())
          - First 5 rows (df.head())

        The `dummy` parameter is unused; pass an empty string or any value.
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
    # Tool 2: list_sheets
    # ------------------------------------------------------------------
    @tool
    def list_sheets(dummy: str = "") -> str:
        """
        List all sheet names available in the Excel workbook.

        Use this tool when the user asks about the structure of the file,
        wants to know how many sheets exist, or mentions a specific sheet by name.

        In the python_repl, all sheets are accessible via the `all_sheets` dict:
            all_sheets['SheetName']  ->  pd.DataFrame

        The `dummy` parameter is unused; pass an empty string or any value.
        """
        names = list(all_sheets.keys())
        lines = [f"  {i + 1}. '{name}' — {len(all_sheets[name])} rows" for i, name in enumerate(names)]
        return "Sheets in workbook:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 3: python_repl
    # ------------------------------------------------------------------
    # Inject df and all_sheets into the REPL namespace.
    repl = PythonAstREPLTool(
        locals={"df": df, "all_sheets": all_sheets, "pd": pd},
        description=(
            "Execute Python / pandas code to analyze the Excel data.\n"
            "Available variables:\n"
            "  df          — primary DataFrame (read-only, do NOT reassign)\n"
            "  all_sheets  — dict mapping sheet name -> DataFrame\n"
            "  pd          — pandas module\n\n"
            "Rules:\n"
            "  - NEVER modify df in place (no df.drop(inplace=True), etc.)\n"
            "  - Use tabulate or .to_markdown() for readable table output\n"
            "  - Keep code concise; one expression per analysis step\n"
            "  - If an operation fails, catch the error and try a different approach\n\n"
            "Example usage:\n"
            "  df.groupby('Category')['Sales'].sum().sort_values(ascending=False)\n"
            "  all_sheets['Clients'].head()"
        ),
    )

    return [get_dataframe_info, list_sheets, repl]


def _pick_engine(path: str) -> str:
    """Select the correct pandas Excel engine based on file extension."""
    return "xlrd" if path.lower().endswith(".xls") else "openpyxl"

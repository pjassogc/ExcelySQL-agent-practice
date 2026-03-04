"""
Solution for Exercise 3 — switch_sheet tool with mutable DataFrameHolder.

This is a complete replacement for tools.py showing how to implement
dynamic sheet switching using a mutable container object.

Teaching point: Python closures capture object *references*, not values.
By wrapping the DataFrame in a mutable object, all tools share the same
reference and see updates immediately.
"""

import io
from typing import Any

import pandas as pd
from langchain.tools import tool
from langchain_experimental.tools import PythonAstREPLTool


class DataFrameHolder:
    """
    Mutable container that lets tools share a reference to the active DataFrame.

    Why not just use a global variable?
    - Globals make testing harder and create implicit coupling.
    - A holder object can be passed explicitly, keeping dependencies clear.

    Why not update the REPL locals dict directly?
    - PythonAstREPLTool's locals dict is set at initialization and is not
      designed to be updated afterwards.
    - The holder pattern avoids re-creating the REPL on each sheet switch.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.current_sheet: str = "default"


def _capture_df_info(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def _pick_engine(path: str) -> str:
    return "xlrd" if path.lower().endswith(".xls") else "openpyxl"


def build_tools(df: pd.DataFrame, excel_path: str) -> list[Any]:
    all_sheets: dict[str, pd.DataFrame] = pd.read_excel(
        excel_path, sheet_name=None, engine=_pick_engine(excel_path)
    )

    # Use the first sheet name as the initial active sheet
    first_sheet = next(iter(all_sheets))
    holder = DataFrameHolder(df)
    holder.current_sheet = first_sheet

    # ------------------------------------------------------------------
    # Tool 1: get_dataframe_info (updated to use holder.df)
    # ------------------------------------------------------------------
    @tool
    def get_dataframe_info(dummy: str = "") -> str:
        """
        Return schema and sample data for the active DataFrame.
        Use this tool FIRST when the user asks a question about the data.
        The `dummy` parameter is unused.
        """
        info = _capture_df_info(holder.df)
        describe = holder.df.describe(include="all").to_string()
        head = holder.df.head(5).to_markdown(index=False)
        return (
            f"Active sheet: '{holder.current_sheet}'\n\n"
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
        List all sheet names in the Excel workbook and show which is active.
        The `dummy` parameter is unused.
        """
        lines = []
        for i, name in enumerate(all_sheets.keys()):
            active_marker = " ← active" if name == holder.current_sheet else ""
            lines.append(f"  {i + 1}. '{name}' — {len(all_sheets[name])} rows{active_marker}")
        return "Sheets in workbook:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 3: switch_sheet (NEW)
    # ------------------------------------------------------------------
    @tool
    def switch_sheet(sheet_name: str) -> str:
        """
        Switch the active DataFrame to a different sheet in the workbook.

        After switching, `df` in the python_repl will refer to the new sheet.
        Use list_sheets first if you're not sure of the exact sheet name.

        Parameters
        ----------
        sheet_name : str
            Name of the sheet to switch to. Case-insensitive matching is attempted.
        """
        # Try exact match first, then case-insensitive
        if sheet_name in all_sheets:
            match = sheet_name
        else:
            match = next(
                (name for name in all_sheets if name.lower() == sheet_name.lower()),
                None,
            )

        if match is None:
            available = ", ".join(f"'{n}'" for n in all_sheets.keys())
            return f"Sheet '{sheet_name}' not found. Available sheets: {available}"

        holder.df = all_sheets[match]
        holder.current_sheet = match
        return (
            f"Switched to sheet '{match}'. "
            f"It has {len(holder.df)} rows and {len(holder.df.columns)} columns.\n"
            f"Columns: {', '.join(holder.df.columns.tolist())}"
        )

    # ------------------------------------------------------------------
    # Tool 4: python_repl
    # Note: we pass `holder` as `df` so the LLM must use `df.df`
    # A better UX: use a proxy object that delegates attribute access.
    # For simplicity here, update the description to explain the usage.
    # ------------------------------------------------------------------
    repl = PythonAstREPLTool(
        locals={"df": holder, "all_sheets": all_sheets, "pd": pd},
        description=(
            "Execute Python / pandas code to analyze the Excel data.\n"
            "Available variables:\n"
            "  df.df       — active DataFrame (use df.df, not df directly)\n"
            "  all_sheets  — dict mapping sheet name -> DataFrame\n"
            "  pd          — pandas module\n\n"
            "IMPORTANT: The active DataFrame is accessed via df.df (not just df).\n"
            "Use switch_sheet tool to change the active sheet.\n\n"
            "Example:\n"
            "  df.df.groupby('Region')['Sales'].sum()\n"
            "  all_sheets['Clients'].head()"
        ),
    )

    return [get_dataframe_info, list_sheets, switch_sheet, repl]


# ---------------------------------------------------------------------------
# Note on SYSTEM_PROMPT addition (agent.py)
# ---------------------------------------------------------------------------
# Add to SYSTEM_PROMPT in agent.py:
#
# AVAILABLE TOOLS:
# - get_dataframe_info: call first to learn the schema
# - list_sheets: lists all sheets with row counts
# - switch_sheet(sheet_name): switches the active DataFrame to another sheet
# - python_repl: runs pandas code; use df.df (not df) for the active sheet

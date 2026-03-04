"""
Solution for Exercise 2 — export_summary tool.

This file shows only the relevant addition to tools.py.
Add the export_summary tool inside build_tools(), before the return statement.
"""

from pathlib import Path

from langchain.tools import tool


# Add this inside build_tools(), before the return statement:

@tool
def export_summary(text: str) -> str:
    """
    Save a text summary to a file called summary.txt in the current directory.

    Use this tool when the user asks to save, export, or write down
    the results of the analysis. Pass the full summary text as the argument.

    Parameters
    ----------
    text : str
        The summary text to save. Can be plain text or markdown.
    """
    output_path = Path("summary.txt").resolve()
    output_path.write_text(text, encoding="utf-8")
    return f"Summary saved to: {output_path}\n\nContent preview:\n{text[:200]}{'...' if len(text) > 200 else ''}"


# Then update the return in build_tools():
# return [get_dataframe_info, list_sheets, repl, export_summary]

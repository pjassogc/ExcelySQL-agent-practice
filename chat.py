"""
chat.py — Entry point for the Excel analyst agent.

Usage
-----
    python chat.py data/sample.xlsx     # pass the file directly
    python chat.py                      # will prompt for file path

The script:
    1. Validates and loads the Excel file into a DataFrame.
    2. Builds tools (get_dataframe_info, list_sheets, python_repl).
    3. Builds the LangGraph ReAct agent with MemorySaver.
    4. Enters a Rich-powered terminal chat loop.

Teaching points:
    - thread_id is fixed per session: all messages share the same checkpoint,
      giving automatic multi-turn memory.
    - We stream the agent output to show intermediate tool calls in real time.
    - Rich provides colour and panels without cluttering the logic.
    - dotenv loads OPENAI_API_KEY transparently so no key appears in code.
"""

import sys
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from agent import build_agent
from tools import build_tools, _pick_engine

# Load environment variables from .env before anything else
load_dotenv()

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".xlsx", ".xls"}


def load_excel(path: str) -> tuple[pd.DataFrame, str]:
    """
    Load the first sheet of an Excel file into a DataFrame.

    Returns
    -------
    (df, resolved_path)
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        console.print(f"[bold red]Error:[/] File not found: {p}")
        sys.exit(1)
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        console.print(f"[bold red]Error:[/] Unsupported format '{p.suffix}'. Use .xlsx or .xls")
        sys.exit(1)

    engine = _pick_engine(str(p))
    df = pd.read_excel(str(p), engine=engine)
    return df, str(p)


def print_welcome(excel_path: str, df: pd.DataFrame) -> None:
    """Print a welcome banner with file information."""
    info = (
        f"[bold cyan]File:[/] {excel_path}\n"
        f"[bold cyan]Rows:[/] {len(df):,}   "
        f"[bold cyan]Columns:[/] {len(df.columns)}\n"
        f"[bold cyan]Columns:[/] {', '.join(df.columns.tolist())}\n\n"
        "[dim]Type your question in plain English. Type [bold]exit[/] or [bold]quit[/] to leave.[/dim]"
    )
    console.print(Panel(info, title="[bold green]Excel Analyst Agent[/]", border_style="green"))


def extract_final_answer(result: dict) -> str:
    """
    Extract the last AI message text from the agent result dict.

    create_react_agent returns {"messages": [...]} where the last message
    is always the final AIMessage.
    """
    messages = result.get("messages", [])
    for msg in reversed(messages):
        # AIMessage has .content; tool messages have different types
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            # Skip messages that are just tool call requests (they have tool_calls)
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return msg.content
    return "(No response generated)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Resolve file path -----------------------------------------------
    if len(sys.argv) > 1:
        excel_path_arg = sys.argv[1]
    else:
        excel_path_arg = Prompt.ask("[bold yellow]Enter path to Excel file[/]")

    df, excel_path = load_excel(excel_path_arg)

    # ---- Build agent -------------------------------------------------------
    with console.status("[bold green]Loading Excel and building agent…[/]"):
        tools = build_tools(df, excel_path)
        graph = build_agent(tools)

    print_welcome(excel_path, df)

    # ---- Chat loop ---------------------------------------------------------
    # A fixed thread_id means all turns share the same MemorySaver checkpoint.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        console.print()
        try:
            user_input = Prompt.ask("[bold blue]You[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        # Invoke the agent
        console.print(Rule("[dim]Agent thinking…[/dim]", style="dim"))
        try:
            result = graph.invoke(
                {"messages": [("human", user_input)]},
                config=config,
            )
        except Exception as exc:
            console.print(f"[bold red]Agent error:[/] {exc}")
            continue

        answer = extract_final_answer(result)
        console.print()
        console.print(Panel(Markdown(answer), title="[bold green]Agent[/]", border_style="green"))


if __name__ == "__main__":
    main()

# chat.py
import platform
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from agent import build_agent
from tools import build_tools, _pick_engine

load_dotenv()

console = Console()
SUPPORTED_EXTENSIONS = {".xlsx", ".xls"}
CHART_PREFIX = "CHART:"


def load_excel(path: str) -> tuple[pd.DataFrame, str]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        console.print(f"[bold red]Error:[/] Archivo no encontrado: {p}")
        sys.exit(1)
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        console.print(f"[bold red]Error:[/] Formato no soportado '{p.suffix}'.")
        sys.exit(1)
    engine = _pick_engine(str(p))
    df = pd.read_excel(str(p), engine=engine)
    return df, str(p)


def print_welcome(excel_path: str, df: pd.DataFrame) -> None:
    info = (
        f"[bold cyan]Archivo:[/] {excel_path}\n"
        f"[bold cyan]Filas:[/] {len(df):,}   "
        f"[bold cyan]Columnas:[/] {len(df.columns)}\n"
        f"[bold cyan]Columnas:[/] {', '.join(df.columns.tolist())}\n\n"
        "[dim]Escribe tu pregunta o pide una gráfica. Escribe [bold]salir[/] para terminar.[/dim]"
    )
    console.print(Panel(info, title="[bold green]Agente Analista de Excel[/]", border_style="green"))


def extract_final_answer(result: dict) -> str:
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            if not getattr(msg, "tool_calls", None):
                return msg.content
    return "(Sin respuesta generada)"


def _handle_charts(result: dict) -> None:
    """Detecta gráficas generadas y las abre con el visor del sistema."""
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage) and msg.content.startswith(CHART_PREFIX):
            path = msg.content[len(CHART_PREFIX):]
            console.print(f"\n[bold yellow]📊 Gráfica guardada:[/] {path}")
            try:
                if platform.system() == "Darwin":
                    subprocess.run(["open", path], check=True)
                elif platform.system() == "Linux":
                    subprocess.run(["xdg-open", path], check=True)
            except Exception:
                pass


def main() -> None:
    if len(sys.argv) > 1:
        excel_path_arg = sys.argv[1]
    else:
        excel_path_arg = Prompt.ask("[bold yellow]Ruta al archivo Excel[/]")

    df, excel_path = load_excel(excel_path_arg)

    with console.status("[bold green]Cargando Excel y construyendo agente…[/]"):
        tools = build_tools(df, excel_path)
        graph = build_agent(tools)

    print_welcome(excel_path, df)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        console.print()
        try:
            user_input = Prompt.ask("[bold blue]Tú[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]¡Hasta luego![/dim]")
            break

        if user_input.strip().lower() in {"salir", "exit", "quit", "q"}:
            console.print("[dim]¡Hasta luego![/dim]")
            break

        if not user_input.strip():
            continue

        console.print(Rule("[dim]Agente pensando…[/dim]", style="dim"))
        try:
            result = graph.invoke(
                {"messages": [("human", user_input)]},
                config=config,
            )
        except Exception as exc:
            console.print(f"[bold red]Error del agente:[/] {exc}")
            continue

        answer = extract_final_answer(result)
        console.print()
        console.print(Panel(Markdown(answer), title="[bold green]Agente[/]", border_style="green"))
        _handle_charts(result)


if __name__ == "__main__":
    main()
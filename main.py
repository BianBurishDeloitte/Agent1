from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent import create_agent
from config import LANGSMITH_PROJECT, LANGSMITH_TRACING, MODEL_NAME
from logger import AgentStepLogger

console = Console()

_EXIT = {"exit", "quit", "q", "bye"}

_HELP = """\
[bold]Commands[/bold]
  [cyan]exit / quit / q[/cyan]  — quit the program
  [cyan]clear[/cyan]            — clear the screen
  [cyan]help[/cyan]             — show this message

[bold]Example prompts[/bold]
  [dim]•[/dim] What is 1337 multiplied by 42?
  [dim]•[/dim] What is the weather in Tel Aviv?
  [dim]•[/dim] Tell me about LangChain
  [dim]•[/dim] What is 25 percent of 840?
  [dim]•[/dim] Search for information about artificial intelligence
  [dim]•[/dim] What is 2 to the power of 10 minus 24?
  [dim]•[/dim] What is the weather in Tokyo, and what is 37 Celsius in Fahrenheit?
"""


def _header() -> None:
    console.clear()
    console.rule("[bold cyan]LangChain ReAct Agent[/bold cyan]", style="cyan")

    # Status table
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", justify="right")
    table.add_column()

    table.add_row("Model", f"[cyan]{MODEL_NAME}[/cyan]")
    table.add_row("Tools", "[cyan]Calculator[/cyan]  [green]Weather[/green]  [yellow]Search[/yellow]")

    if LANGSMITH_TRACING:
        table.add_row(
            "LangSmith",
            f"[bold green]● Tracing ON[/bold green]  "
            f"[dim]project:[/dim] [green]{LANGSMITH_PROJECT}[/green]  "
            f"[dim]→ smith.langchain.com[/dim]",
        )
    else:
        table.add_row(
            "LangSmith",
            "[dim]○ Tracing OFF[/dim]  [dim](set LANGCHAIN_TRACING_V2=true in .env)[/dim]",
        )

    console.print(
        Panel(
            table,
            title="[bold cyan]Welcome[/bold cyan]",
            border_style="cyan",
            box=box.DOUBLE,
        )
    )
    console.print()
    console.print("Type [bold]help[/bold] for examples or [bold]exit[/bold] to quit.\n")


def main() -> None:
    _header()
    agent = create_agent()
    logger = AgentStepLogger()

    while True:
        try:
            console.print("[bold cyan]You ›[/bold cyan] ", end="")
            user_input = input().strip()

            if not user_input:
                continue
            if user_input.lower() in _EXIT:
                console.print("\n[bold cyan]Goodbye![/bold cyan]")
                break
            if user_input.lower() == "help":
                console.print(Panel(_HELP, title="Help", border_style="white", box=box.ROUNDED))
                continue
            if user_input.lower() == "clear":
                _header()
                continue

            console.print()
            console.rule("[bold white]Processing[/bold white]", style="white dim")
            console.print(
                Panel(
                    user_input,
                    title="[bold white]Query[/bold white]",
                    border_style="white",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )

            agent.invoke(
                {"input": user_input},
                config={"callbacks": [logger]},
            )

        except KeyboardInterrupt:
            console.print("\n\n[bold cyan]Interrupted — goodbye![/bold cyan]")
            break
        except Exception as exc:
            console.print(
                Panel(
                    str(exc),
                    title="[bold red]Unexpected Error[/bold red]",
                    border_style="red",
                    box=box.ROUNDED,
                )
            )


if __name__ == "__main__":
    main()

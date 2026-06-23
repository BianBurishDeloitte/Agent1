import json

from langchain_core.messages import AIMessage, ToolMessage
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent import create_agent
from config import LANGSMITH_PROJECT, LANGSMITH_TRACING, MODEL_NAME

console = Console()

_EXIT = {"exit", "quit", "q", "bye"}

_HELP = """\
[bold]Commands[/bold]
  [cyan]exit / quit / q[/cyan]  — quit
  [cyan]clear[/cyan]            — clear the screen
  [cyan]help[/cyan]             — show this message

[bold]Example prompts[/bold]
  [dim]•[/dim] What is 1337 multiplied by 42?
  [dim]•[/dim] What is the weather in Tel Aviv?
  [dim]•[/dim] Tell me about LangChain
  [dim]•[/dim] What is 25 percent of 840?
  [dim]•[/dim] What is the weather in Tokyo, and what is 37 Celsius in Fahrenheit?
"""


def _header() -> None:
    console.clear()
    console.rule("[bold cyan]Agent Team Meeting — LangChain ReAct Agent[/bold cyan]", style="cyan")

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", justify="right")
    table.add_column()
    table.add_row("Model",   f"[cyan]{MODEL_NAME}[/cyan]")
    table.add_row("Tools",   "[cyan]Calculator[/cyan]  [green]Weather[/green]  [yellow]Search[/yellow]")
    table.add_row(
        "LangSmith",
        (f"[bold green]● ON[/bold green]  project: [green]{LANGSMITH_PROJECT}[/green]  → smith.langchain.com"
         if LANGSMITH_TRACING else
         "[dim]○ OFF  (set LANGCHAIN_TRACING_V2=true in .env)[/dim]"),
    )

    console.print(Panel(table, title="[bold cyan]Welcome[/bold cyan]", border_style="cyan", box=box.DOUBLE))
    console.print()
    console.print("Type [bold]help[/bold] for examples or [bold]exit[/bold] to quit.\n")


def _run(agent, user_input: str) -> None:
    """Stream the agent and display each step with Rich formatting."""
    console.print()
    console.rule("[bold white]Processing[/bold white]", style="white dim")
    console.print(
        Panel(user_input, title="[bold white]Query[/bold white]",
              border_style="white", box=box.ROUNDED, padding=(0, 1))
    )

    step = 0
    final_answer = ""

    for chunk in agent.stream(
        {"messages": [("human", user_input)]},
        stream_mode="updates",
    ):
        for node_name, node_output in chunk.items():
            for msg in node_output.get("messages", []):

                # ── Tool call → Action ────────────────────────────────────
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        step += 1
                        console.print()
                        console.rule(f"[bold yellow]  Step {step}  [/bold yellow]", style="yellow dim")
                        args = (json.dumps(tc["args"], ensure_ascii=False)
                                if isinstance(tc["args"], dict) else str(tc["args"]))
                        from rich.text import Text
                        body = Text()
                        body.append("Tool  : ", style="bold dim")
                        body.append(tc["name"], style="bold blue")
                        body.append("\nInput : ", style="bold dim")
                        body.append(args, style="italic cyan")
                        console.print(
                            Panel(body, title="[bold blue]Action[/bold blue]",
                                  border_style="blue", box=box.ROUNDED, padding=(0, 1))
                        )

                # ── Tool result → Observation ─────────────────────────────
                elif isinstance(msg, ToolMessage):
                    console.print(
                        Panel(str(msg.content), title="[bold green]Observation[/bold green]",
                              border_style="green", box=box.ROUNDED, padding=(0, 1))
                    )

                # ── Final text with no tool call → Final Answer ───────────
                elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_answer = msg.content

    if final_answer:
        console.print()
        console.print(
            Panel(final_answer, title="[bold cyan]  Final Answer  [/bold cyan]",
                  border_style="cyan", box=box.DOUBLE, padding=(1, 2))
        )
        console.rule(style="cyan dim")


def main() -> None:
    _header()
    agent = create_agent()

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

            _run(agent, user_input)

        except KeyboardInterrupt:
            console.print("\n\n[bold cyan]Interrupted — goodbye![/bold cyan]")
            break
        except Exception as exc:
            console.print(
                Panel(str(exc), title="[bold red]Error[/bold red]",
                      border_style="red", box=box.ROUNDED)
            )


if __name__ == "__main__":
    main()

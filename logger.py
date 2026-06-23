from langchain_core.callbacks.base import BaseCallbackHandler
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()


class AgentStepLogger(BaseCallbackHandler):
    """Rich-formatted callback handler that visualises every ReAct step."""

    def __init__(self) -> None:
        super().__init__()
        self._step = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_thought(self, log: str) -> str:
        """Pull the Thought text out of the raw LLM output."""
        log = log.strip()
        # The ReAct log looks like: "Thought: ...\nAction: ...\nAction Input: ..."
        if "Action:" in log:
            raw = log.split("\nAction:")[0]
        else:
            raw = log
        return raw.replace("Thought:", "").strip()

    # ------------------------------------------------------------------
    # Callback methods
    # ------------------------------------------------------------------

    def on_agent_action(self, action, **kwargs) -> None:
        self._step += 1
        thought = self._extract_thought(action.log)

        console.print()
        console.rule(
            f"[bold yellow]  Step {self._step}  [/bold yellow]",
            style="yellow dim",
        )

        if thought:
            console.print(
                Panel(
                    thought,
                    title="[bold yellow]Thought[/bold yellow]",
                    border_style="yellow",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )

        action_body = Text()
        action_body.append("Tool  : ", style="bold dim")
        action_body.append(action.tool, style="bold blue")
        action_body.append("\nInput : ", style="bold dim")
        action_body.append(str(action.tool_input), style="italic cyan")

        console.print(
            Panel(
                action_body,
                title="[bold blue]Action[/bold blue]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def on_tool_end(self, output, **kwargs) -> None:
        console.print(
            Panel(
                str(output),
                title="[bold green]Observation[/bold green]",
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def on_tool_error(self, error, **kwargs) -> None:
        console.print(
            Panel(
                str(error),
                title="[bold red]Tool Error[/bold red]",
                border_style="red",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def on_agent_finish(self, finish, **kwargs) -> None:
        output = finish.return_values.get("output", "")
        console.print()
        console.print(
            Panel(
                output,
                title="[bold cyan]  Final Answer  [/bold cyan]",
                border_style="cyan",
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )
        console.rule(style="cyan dim")
        self._step = 0  # reset for next invocation

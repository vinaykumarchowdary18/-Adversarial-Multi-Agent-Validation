"""
main.py — AMAV command-line entry point.

Usage:
  python main.py "Your question or task here"
  python main.py --task "Summarise recent advances in CRISPR gene editing (2024-2025)"
  python main.py --interactive          # REPL mode

The script:
  1. Loads config (fails fast if keys missing).
  2. Runs the orchestrator debate loop.
  3. Saves output to ./outputs/ as both JSON and Markdown.
  4. Prints the final answer + confidence to stdout.
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

from core.config import load_config
from core.orchestrator import Orchestrator
from core.models import FinalAnswer
from core.logger import get_logger
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

log = get_logger(__name__)
console = Console()


def _save_outputs(answer: FinalAnswer, outputs_dir: str) -> tuple[Path, Path]:
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = answer.task[:40].replace(" ", "_").replace("/", "-")
    slug = "".join(c for c in slug if c.isalnum() or c in ("_", "-"))

    json_path = out / f"{ts}_{slug}.json"
    md_path   = out / f"{ts}_{slug}.md"

    # JSON — full machine-readable record
    json_path.write_text(answer.model_dump_json(indent=2), encoding="utf-8")

    # Markdown — human-readable report
    md_lines = [
        f"# AMAV Answer\n",
        f"**Task:** {answer.task}\n",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n",
        f"**Confidence:** {answer.confidence:.0%}  |  "
        f"**Consensus score:** {answer.consensus_score:.2f}  |  "
        f"**Debate rounds:** {answer.debate_rounds}\n",
        f"\n---\n",
        f"## Answer\n",
        answer.answer,
        f"\n---\n",
    ]

    if answer.sources:
        md_lines += ["\n## Sources\n"]
        md_lines += [f"- {url}\n" for url in answer.sources]

    if answer.dissenting_points:
        md_lines += ["\n## Unresolved dissents (minority critique)\n"]
        md_lines += [f"- {pt}\n" for pt in answer.dissenting_points]

    md_lines += ["\n## Models used\n"]
    for role, model in answer.models_used.items():
        md_lines += [f"- **{role}**: `{model}`\n"]

    md_path.write_text("".join(md_lines), encoding="utf-8")

    return json_path, md_path


def _print_answer(answer: FinalAnswer) -> None:
    """Pretty-print the final answer to the terminal."""
    confidence_color = (
        "green" if answer.confidence >= 0.75
        else "yellow" if answer.confidence >= 0.5
        else "red"
    )
    header = (
        f"[{confidence_color}]Confidence: {answer.confidence:.0%}[/{confidence_color}]"
        f"  |  Consensus: {answer.consensus_score:.2f}"
        f"  |  Rounds: {answer.debate_rounds}"
        f"  |  Sources: {len(answer.sources)}"
    )
    console.print()
    console.print(Panel(Markdown(answer.answer), title=header, border_style="cyan"))

    if answer.dissenting_points:
        console.print(
            "\n[yellow]⚠ Unresolved minority critique points:[/yellow]"
        )
        for pt in answer.dissenting_points:
            console.print(f"  • {pt}")
    console.print()


async def _run_once(task: str, config) -> FinalAnswer:
    orchestrator = Orchestrator(config)
    return await orchestrator.run(task)


async def interactive_loop(config) -> None:
    console.print(
        Panel(
            "[bold cyan]AMAV Interactive Mode[/bold cyan]\n"
            "Type your question and press Enter. Type [bold]exit[/bold] or [bold]quit[/bold] to stop.",
            border_style="cyan",
        )
    )
    while True:
        try:
            console.print("\n[bold cyan]>[/bold cyan] ", end="")
            task = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not task:
            continue
        if task.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        try:
            answer = await _run_once(task, config)
            _print_answer(answer)
            json_path, md_path = _save_outputs(answer, config.outputs_dir)
            console.print(f"[dim]Saved → {md_path}[/dim]")
        except Exception as exc:
            log.exception(f"Pipeline error: {exc}")
            console.print(f"[red]Error:[/red] {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="amav",
        description="Adversarial Multi-Agent Validation — personal AI with cross-validating debate",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("task", nargs="?", help="Task / question to answer")
    group.add_argument("--task", "-t", dest="task_flag", help="Task / question (flag form)")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive REPL session",
    )
    parser.add_argument(
        "--output", "-o",
        help="Override output directory (default: from .env / ./outputs)",
    )
    args = parser.parse_args()

    # Load config (crashes loudly if keys missing)
    try:
        config = load_config()
    except EnvironmentError as e:
        console.print(f"[red]Config error:[/red] {e}")
        sys.exit(1)

    # Override output dir if passed
    if args.output:
        import os
        os.environ["OUTPUTS_DIR"] = args.output
        config = load_config()

    task = args.task or args.task_flag

    if args.interactive or not task:
        asyncio.run(interactive_loop(config))
    else:
        try:
            answer = asyncio.run(_run_once(task, config))
        except Exception as exc:
            log.exception(f"Pipeline error: {exc}")
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)

        _print_answer(answer)

        json_path, md_path = _save_outputs(answer, config.outputs_dir)
        console.print(f"[dim]Report saved → {md_path}[/dim]")
        console.print(f"[dim]JSON saved   → {json_path}[/dim]")


if __name__ == "__main__":
    main()

"""
tools/list_github_models.py — discover which GitHub Models your token can access.

Usage:
  python tools/list_github_models.py
  python tools/list_github_models.py --filter gpt
  python tools/list_github_models.py --filter openai

Requires GITHUB_TOKEN in your .env with Models: read scope.
"""
import asyncio
import argparse
import httpx
from dotenv import load_dotenv
import os
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()


async def list_models(filter_str: str | None = None) -> None:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        console.print("[red]GITHUB_TOKEN not set in .env[/red]")
        return

    url = "https://models.github.ai/catalog/models"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2026-03-10",
        "Accept": "application/vnd.github+json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            console.print(f"[red]HTTP {resp.status_code}:[/red] {resp.text[:300]}")
            return

    models = resp.json()
    if not isinstance(models, list):
        # Some endpoints wrap in {"models": [...]}
        models = models.get("models", models.get("data", []))

    table = Table(title="GitHub Models available to your token", show_lines=True)
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Display name", style="white")
    table.add_column("Publisher", style="green")
    table.add_column("Task", style="dim")

    count = 0
    for m in models:
        model_id   = m.get("id", m.get("name", "?"))
        name       = m.get("displayName", m.get("display_name", ""))
        publisher  = m.get("publisher", m.get("org", ""))
        task       = m.get("task", m.get("model_type", ""))

        if filter_str and filter_str.lower() not in (model_id + name + publisher).lower():
            continue

        table.add_row(model_id, name, publisher, task)
        count += 1

    console.print(table)
    console.print(f"\n[dim]{count} models listed{'(filtered)' if filter_str else ''}.[/dim]")
    console.print(
        "\n[bold]To use one:[/bold] set [cyan]GITHUB_MODEL=<model_id>[/cyan] in your .env"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="List GitHub Models available to your token")
    parser.add_argument("--filter", "-f", help="Filter by model ID / name substring")
    args = parser.parse_args()
    asyncio.run(list_models(args.filter))


if __name__ == "__main__":
    main()

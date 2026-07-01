"""The command-line interface: interactive prompts and Rich output over the public API.

Every command gathers input (interactively or from flags) and calls the library
functions in the other modules. The interactive and terminal-rendering code lives only
here, so the rest of the package stays importable and UI-free.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
import questionary
from dotenv import load_dotenv
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress as RichProgress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

load_dotenv()

from prompt2dataset.classify import SUPPORTED_MODELS, crossval, train
from prompt2dataset.clean import apply_flags, find_exact_duplicates, find_outliers
from prompt2dataset.models import Dataset, ReviewStatus
from prompt2dataset.paths import manifest_path, meta_dir
from prompt2dataset.pipeline import generate
from prompt2dataset.progress import Progress
from prompt2dataset.resolver import resolve_subjects
from prompt2dataset.sources import REGISTRY
from prompt2dataset.store import load_dataset, save_dataset

console = Console()
log = logging.getLogger(__name__)

STYLE = questionary.Style([
    ("qmark", "fg:#a78bfa bold"),
    ("question", "bold"),
    ("answer", "fg:#34d399 bold"),
    ("pointer", "fg:#a78bfa bold"),
    ("highlighted", "fg:#a78bfa bold"),
    ("selected", "fg:#34d399"),
])


def _open_folder(path: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(str(path.resolve()))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path.resolve())], check=False)
        else:
            subprocess.run(["xdg-open", str(path.resolve())], check=False)
    except Exception:
        pass


def _load_subjects_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    raw = json.loads(text) if text.startswith("[") else text.splitlines()
    return [s.strip() for s in raw if isinstance(s, str) and s.strip()]


def _bar(description: str):
    """A Rich progress bar wired to a library on_progress callback.

    Returns (callback, context-manager). The callback advances a single task the bar
    creates on the first call, so the library drives the display without knowing Rich.
    """
    progress = RichProgress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    state = {"task": None, "total": None}

    def cb(p: Progress) -> None:
        if state["task"] is None or state["total"] != p.total:
            if state["task"] is not None:
                progress.remove_task(state["task"])
            state["task"] = progress.add_task(p.message or description, total=p.total)
            state["total"] = p.total
        progress.update(state["task"], completed=p.done, description=p.message or description)

    return cb, progress


@click.group()
@click.option("--debug", is_flag=True, help="Enable verbose logging")
def cli(debug: bool) -> None:
    """prompt2dataset: build image datasets from a prompt."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.CRITICAL,
        format="%(levelname)s %(name)s: %(message)s",
    )


@cli.command()
@click.option("--subjects", "subjects_file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Read subjects from a file instead of resolving the prompt.")
def add(subjects_file: Optional[Path]) -> None:
    """Populate the current directory with images."""
    root = Path.cwd()
    console.print(Rule(f"[bold violet]prompt2dataset[/] [dim]{root.name}[/]"))

    existing = load_dataset(root) if manifest_path(root).exists() else None
    if existing:
        prompt = existing.prompt
        console.print(f"[dim]Prompt:[/] {prompt}\n")
    else:
        prompt = (questionary.text("What dataset do you want to build?", style=STYLE).ask() or "").strip()
        if not prompt:
            return

    chosen = questionary.checkbox(
        "Select one or more sources:",
        choices=[questionary.Choice(f"{n}  -  {a.description[:60]}", value=n, checked=(n == "duckduckgo"))
                 for n, a in REGISTRY.items()],
        style=STYLE,
    ).ask()
    if not chosen:
        console.print("[yellow]No sources selected.[/]")
        return

    limit = int(questionary.text("How many images per subject per source?", default="20",
                                 validate=lambda v: v.isdigit() and int(v) > 0 or "positive integer",
                                 style=STYLE).ask() or "20")

    if subjects_file:
        subjects = _load_subjects_file(subjects_file)
    else:
        with console.status("[bold]Resolving subjects...[/]", spinner="dots"):
            subjects = resolve_subjects(prompt)
    if not subjects:
        console.print("[red]Empty subject list.[/]")
        return

    console.print(Columns([f"  [dim]{s}[/]" for s in subjects], equal=True))
    if not questionary.confirm("Fetch images for these subjects?", default=True, style=STYLE).ask():
        return

    ds = existing or Dataset(dataset_id=root.name, prompt=prompt, subjects=[], sources=[])
    cb, bar = _bar("Working")
    with bar:
        result = generate(ds, root, subjects, chosen, limit, on_progress=cb)

    console.print(f"\n[green]v[/] {result.saved} images saved" +
                  (f", [red]{result.failed} failed[/]" if result.failed else "") +
                  (f", {result.dropped} pruned" if result.dropped else ""))
    _summary(ds, root)


def _summary(ds: Dataset, root: Path) -> None:
    stats = ds.stats()
    console.print()
    console.print(Rule("[bold green]Collection complete[/]"))
    grid = Table.grid(padding=(0, 4))
    for _ in range(3):
        grid.add_column(justify="center")
    grid.add_row(f"[bold]{stats['total']}[/]\n[dim]images[/]",
                 f"[bold]{len(ds.subjects)}[/]\n[dim]subjects[/]",
                 f"[bold]{len(ds.sources)}[/]\n[dim]sources[/]")
    console.print(grid, justify="center")
    console.print(f"\n[dim]Saved to[/] [cyan]{root.resolve()}[/]")
    _open_folder(root)


@cli.command()
@click.option("--misclassified", is_flag=True, help="Only review images a model got wrong.")
def review(misclassified: bool) -> None:
    """Interactively review pending items. A accept, D delete, S skip, Q quit."""
    root = Path.cwd()
    ds = load_dataset(root)
    pending = ds.pending_review()

    if misclassified:
        mc_file = meta_dir(root) / "misclassified.json"
        if not mc_file.exists():
            raise click.ClickException("No misclassified.json. Run `p2d train` or `p2d crossval` first.")
        paths = {e["path"] for e in json.loads(mc_file.read_text(encoding="utf-8"))}
        pending = [i for i in pending if str((root / i.local_path).resolve()) in paths]

    if not pending:
        console.print("[green]No items pending review.[/]")
        return

    for idx, item in enumerate(pending, 1):
        console.print(Panel(f"[bold]{item.subject or item.label}[/]  [dim]{item.item_id}[/]\n{item.source_url}",
                            title=f"{idx}/{len(pending)}", border_style="blue"))
        while True:
            key = click.prompt("  [A]ccept [D]elete [S]kip [Q]uit", default="s").strip().lower()
            if key in ("a", "accept"):
                item.review_status = ReviewStatus.valid
                break
            if key in ("d", "delete"):
                p = root / item.local_path
                if p.exists():
                    p.unlink()
                ds.items = [i for i in ds.items if i.item_id != item.item_id]
                break
            if key in ("s", "skip", ""):
                break
            if key in ("q", "quit"):
                save_dataset(ds, root)
                return
    save_dataset(ds, root)


@cli.command()
def info() -> None:
    """Print a summary of the dataset in the current directory."""
    root = Path.cwd()
    ds = load_dataset(root)
    stats = ds.stats()
    table = Table(title=f"Dataset: {ds.dataset_id}", show_lines=True)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Prompt", ds.prompt)
    table.add_row("Sources", ", ".join(ds.sources))
    table.add_row("Subjects", str(len(ds.subjects)))
    for k in ("total", "pending", "valid", "invalid"):
        table.add_row(k.capitalize(), str(stats[k]))
    console.print(table)


def _candidates(ds: Dataset, root: Path):
    return [i for i in ds.items if i.review_status != ReviewStatus.invalid
            and (root / i.local_path).exists()]


@cli.command(name="dedup")
@click.option("--delete", is_flag=True, help="Delete flagged files instead of marking invalid.")
def dedup_cmd(delete: bool) -> None:
    """Remove exact-duplicate images."""
    root = Path.cwd()
    ds = load_dataset(root)
    with console.status("[bold]Hashing pixels...[/]"):
        dupes = find_exact_duplicates(_candidates(ds, root), root)
    if not dupes:
        console.print("[green]No duplicates found.[/]")
        return
    apply_flags(dupes, ds, root, delete=delete)
    save_dataset(ds, root)
    console.print(f"[green]v[/] {'Deleted' if delete else 'Flagged'} {len(dupes)} duplicates")


@cli.command(name="outliers")
@click.option("--eps", default=0.25, show_default=True, help="DBSCAN cosine radius. Lower flags more.")
@click.option("--delete", is_flag=True, help="Delete flagged files instead of marking invalid.")
def outliers_cmd(eps: float, delete: bool) -> None:
    """Remove outlier images that don't fit the rest of their subject."""
    root = Path.cwd()
    ds = load_dataset(root)
    try:
        with console.status("[bold]Embedding images...[/]"):
            flagged = find_outliers(_candidates(ds, root), root, eps)
    except ImportError as exc:
        raise click.ClickException(f'Needs the [train] extra ({exc.name} missing). '
                                   'Install: pip install "prompt2dataset[train]".')
    if not flagged:
        console.print("[green]No outliers found.[/]")
        return
    apply_flags(flagged, ds, root, delete=delete)
    save_dataset(ds, root)
    console.print(f"[green]v[/] {'Deleted' if delete else 'Flagged'} {len(flagged)} outliers")


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        raise click.ClickException('Needs the [train] extra. Install: pip install "prompt2dataset[train]".')


@cli.command(name="train")
@click.option("--epochs", default=5, show_default=True)
@click.option("--val-split", default=0.2, show_default=True)
@click.option("--img-size", default=224, show_default=True)
@click.option("--model", "model_name", default="mobilenet_v2", show_default=True, type=click.Choice(SUPPORTED_MODELS))
def train_cmd(epochs: int, val_split: float, img_size: int, model_name: str) -> None:
    """Fine-tune a pretrained image classifier on the current dataset."""
    _require_torch()
    root = Path.cwd()
    ds = load_dataset(root)
    if not ds.items:
        raise click.ClickException("No items. Run `p2d add` first.")
    console.print(Rule(f"[bold violet]p2d train[/] [dim]{root.name}[/]"))
    cb, bar = _bar("Training")
    with bar:
        report = train(root, ds.items, model=model_name, epochs=epochs, val_split=val_split,
                       img_size=img_size, on_progress=cb)
    console.print(Panel(f"[bold]Overall accuracy:[/] {report['overall_accuracy']:.1%}",
                        title="[green]Training complete[/]", border_style="green"))


@cli.command(name="crossval")
@click.option("--folds", default=5, show_default=True)
@click.option("--epochs", default=5, show_default=True)
@click.option("--img-size", default=224, show_default=True)
@click.option("--seed", default=None, type=int, help="Seed the fold shuffle for reproducible runs.")
def crossval_cmd(folds: int, epochs: int, img_size: int, seed: Optional[int]) -> None:
    """Find likely mislabeled images by out-of-fold cross-validation."""
    _require_torch()
    root = Path.cwd()
    ds = load_dataset(root)
    if not ds.items:
        raise click.ClickException("No items. Run `p2d add` first.")
    console.print(Rule(f"[bold violet]p2d crossval[/] [dim]{root.name}[/]"))
    cb, bar = _bar("Cross-validating")
    with bar:
        mis = crossval(root, ds.items, folds=folds, epochs=epochs, img_size=img_size, seed=seed, on_progress=cb)
    console.print(Panel(f"[bold]{len(mis)}[/] likely mislabeled images.\n"
                        f"Run [bold]p2d review --misclassified[/] to step through them.",
                        title="[green]Cross-validation complete[/]", border_style="green"))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

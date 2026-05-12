from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import click
import questionary
from dotenv import load_dotenv, set_key
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

load_dotenv()

from prompt2dataset.train import train_cmd
from prompt2dataset.models import Dataset, DatasetItem, ReviewStatus
from prompt2dataset.resolver import resolve_subjects
from prompt2dataset.sources import REGISTRY, fetch_all
from prompt2dataset.utils import meta_dir

console = Console()
log = logging.getLogger(__name__)

ENV_FILE = Path(".env")
DOWNLOAD_RATE_LIMIT = 0.1

STYLE = questionary.Style([
    ("qmark",       "fg:#a78bfa bold"),
    ("question",    "bold"),
    ("answer",      "fg:#34d399 bold"),
    ("pointer",     "fg:#a78bfa bold"),
    ("highlighted", "fg:#a78bfa bold"),
    ("selected",    "fg:#34d399"),
    ("separator",   "fg:#6b7280"),
    ("instruction", "fg:#6b7280"),
])


def _key_is_set() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return bool(key) and not key.startswith("sk-ant-...")


def _ensure_api_key() -> None:
    if _key_is_set():
        return

    console.print()
    console.print(Panel(
        "An Anthropic API key is required.\n"
        "Get yours at [cyan]https://console.anthropic.com/[/]\n\n"
        "The key will be saved to [dim].env[/] in this directory.",
        title="[yellow]API key not found[/]",
        border_style="yellow",
    ))
    console.print()

    key = questionary.password("Paste your Anthropic API key:", style=STYLE).ask()
    if not key or not key.strip():
        raise click.ClickException("No API key provided.")

    key = key.strip()
    os.environ["ANTHROPIC_API_KEY"] = key
    set_key(str(ENV_FILE), "ANTHROPIC_API_KEY", key)
    console.print("[green]v[/] Key saved to .env\n")


def load_dataset(dataset_root: Path) -> Dataset:
    path = meta_dir(dataset_root) / "manifest.json"
    if not path.exists():
        raise click.ClickException(f"No manifest found at {path}")
    return Dataset.model_validate_json(path.read_text(encoding="utf-8"))


def save_dataset(ds: Dataset, dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    md = meta_dir(dataset_root)
    (md / "manifest.json").write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    _write_labels(ds, md)


def _write_labels(ds: Dataset, md: Path) -> None:
    lines = ["filename,subject,source"]
    for item in ds.items:
        source = item.meta.get("source", "unknown")
        lines.append(f"{item.local_path},{item.label},{source}")
    (md / "labels.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:80]


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


def _extension_for(url: str) -> str:
    ext = Path(url.split("?")[0].rstrip("/")).suffix.lower()
    return ext if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp"} else ".jpg"


def _download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    contact = os.environ.get("P2D_CONTACT", "unknown")
    ua = f"prompt2dataset/0.1 ({contact}) urllib/3"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=20) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as exc:
        log.warning("Download failed %s: %s", url, exc)
        return False


def _records_to_items(raw_results: dict) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for subject, source_map in raw_results.items():
        label = _slug(subject)
        for records in source_map.values():
            for rec in records:
                url = rec.get("url", "")
                if not url:
                    continue
                item_id = DatasetItem.make_id(url)
                ext = _extension_for(url)
                items.append(DatasetItem(
                    item_id=item_id,
                    label=label,
                    source_url=url,
                    local_path=str(Path(label) / f"{label}_{item_id}{ext}"),
                    meta={k: v for k, v in rec.items() if k != "url"},
                ))
    return items


def _download_items(items: list[DatasetItem], dataset_root: Path) -> tuple[int, int]:
    ok = failed = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="yellow", complete_style="yellow"),
        TextColumn("[cyan]{task.completed}[/]/[cyan]{task.total}[/] files"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("  Saving images", total=len(items))
        for item in items:
            dest = dataset_root / item.local_path
            if dest.exists():
                ok += 1
                progress.advance(task)
                continue
            if _download_file(item.source_url, dest):
                ok += 1
            else:
                failed += 1
            progress.advance(task)
            time.sleep(DOWNLOAD_RATE_LIMIT)
    return ok, failed


def _run_add(dataset_root: Path) -> Optional[Dataset]:
    console.print()
    console.print(Rule(f"[bold violet]prompt2dataset[/] [dim]{dataset_root.name}[/]"))
    console.print()

    existing_ds: Optional[Dataset] = None

    if (meta_dir(dataset_root) / "manifest.json").exists():
        existing_ds = load_dataset(dataset_root)
        prompt = existing_ds.prompt
        console.print(f"[dim]Prompt:[/] {prompt}\n")
    else:
        prompt = questionary.text(
            "What image dataset do you want to build?",
            style=STYLE,
        ).ask()
        if not prompt:
            return None
        prompt = prompt.strip()

    cap_str = questionary.text(
        "Max subjects to use? (leave blank for all)",
        default="",
        validate=lambda v: v == "" or (v.isdigit() and int(v) > 0) or "Enter a positive integer or leave blank",
        style=STYLE,
    ).ask()
    if cap_str is None:
        return None
    subject_cap = int(cap_str) if cap_str.strip() else None

    chosen = questionary.checkbox(
        "Select one or more sources:",
        choices=[
            questionary.Choice(
                f"{name}  -  {a.description[:65]}...",
                value=name,
                checked=(name == "duckduckgo"),
            )
            for name, a in REGISTRY.items()
        ],
        style=STYLE,
    ).ask()
    if not chosen:
        console.print("[yellow]No sources selected, aborted.[/]")
        return None
    manual_sources = chosen

    limit_str = questionary.text(
        "How many images per subject per source?",
        default="20",
        validate=lambda v: v.isdigit() and int(v) > 0 or "Enter a positive integer",
        style=STYLE,
    ).ask()
    if limit_str is None:
        return None
    limit = int(limit_str)

    console.print()

    with console.status("[bold]Resolving subjects...[/]", spinner="dots"):
        try:
            all_subjects = resolve_subjects(prompt)
        except Exception as exc:
            console.print(f"[red]Subject resolution failed:[/] {exc}")
            return None

    if not all_subjects:
        console.print("[red]Claude returned an empty subject list.[/]")
        return None

    if subject_cap and len(all_subjects) > subject_cap:
        all_subjects = all_subjects[:subject_cap]

    # Persist immediately so a crash or abort after this point doesn't lose the list
    dataset_root.mkdir(parents=True, exist_ok=True)
    (meta_dir(dataset_root) / "subjects.json").write_text(
        json.dumps(all_subjects, indent=2), encoding="utf-8"
    )

    existing_subjects: set[str] = set(existing_ds.subjects) if existing_ds else set()
    new_subjects = [s for s in all_subjects if s not in existing_subjects]

    if existing_ds and not new_subjects:
        console.print(
            f"[green]v[/] All {len(all_subjects)} subjects already in dataset, nothing new to fetch.\n"
        )
        return existing_ds

    if existing_subjects:
        console.print(
            f"[green]v[/] [bold]{len(new_subjects)} new subjects[/] "
            f"[dim]({len(existing_subjects)} already present, skipped)[/]"
        )
    else:
        console.print(f"[green]v[/] [bold]{len(new_subjects)} subjects[/] identified")

    console.print(Columns([f"  [dim]{s}[/]" for s in new_subjects], equal=True, expand=False))
    console.print()

    keep = questionary.confirm("Keep subjects?", default=True, style=STYLE).ask()
    if keep is None:
        return None
    if not keep:
        kept = questionary.checkbox(
            "Select subjects to keep:",
            choices=[questionary.Choice(s, value=s, checked=True) for s in new_subjects],
            style=STYLE,
        ).ask()
        if kept is None:
            return None
        if not kept:
            console.print("[yellow]No subjects selected, aborted.[/]")
            return None
        new_subjects = kept
        console.print(f"[dim]  {len(new_subjects)} subjects kept[/]\n")

    source_list = manual_sources
    console.print(f"[green]v[/] [bold]Sources:[/] {', '.join(source_list)}")
    console.print()

    confirmed = questionary.confirm(
        f"Fetch {limit} images x {len(new_subjects)} subjects x {len(source_list)} source(s)?",
        default=True,
        style=STYLE,
    ).ask()
    if not confirmed:
        return None

    console.print()

    if existing_ds:
        ds = existing_ds
        ds.subjects += new_subjects
        for s in source_list:
            if s not in ds.sources:
                ds.sources.append(s)
    else:
        ds = Dataset(
            dataset_id=dataset_root.name,
            prompt=prompt,
            subjects=new_subjects,
            sources=source_list,
        )

    for subject in new_subjects:
        (dataset_root / _slug(subject)).mkdir(exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.completed}[/]/[cyan]{task.total}[/] subjects"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"  Querying {', '.join(source_list)}",
            total=len(new_subjects),
        )

        async def _fetch_with_progress() -> dict:
            results: dict = {}
            for subject in new_subjects:
                partial = await fetch_all([subject], source_list, limit)
                results.update(partial)
                progress.advance(task)
            return results

        raw_results = asyncio.run(_fetch_with_progress())

    total_records = sum(
        len(recs) for src_map in raw_results.values() for recs in src_map.values()
    )
    console.print(f"[green]v[/] [bold]{total_records}[/] records retrieved\n")

    if total_records == 0:
        console.print("[yellow]No records found. Try a different prompt or sources.[/]")
        save_dataset(ds, dataset_root)
        return ds

    new_items = _records_to_items(raw_results)
    added = ds.add_items(new_items)
    skipped = len(new_items) - added
    if skipped:
        console.print(f"[dim]  {skipped} duplicates skipped[/]")

    pending = [i for i in ds.items if not (dataset_root / i.local_path).exists()]
    if pending:
        ok, failed = _download_items(pending, dataset_root)
        line = f"[green]v[/] {ok} images saved"
        if failed:
            line += f", [red]{failed} failed[/]"
        console.print(line)

    save_dataset(ds, dataset_root)
    return ds


def _print_summary(ds: Dataset, dataset_root: Path) -> None:
    stats = ds.stats()

    console.print()
    console.print(Rule("[bold green]Collection complete[/]"))
    console.print()

    grid = Table.grid(padding=(0, 4))
    for _ in range(3):
        grid.add_column(justify="center")
    grid.add_row(
        f"[bold]{stats['total']}[/]\n[dim]images[/]",
        f"[bold]{len(ds.subjects)}[/]\n[dim]subjects[/]",
        f"[bold]{len(ds.sources)}[/]\n[dim]sources[/]",
    )
    console.print(grid, justify="center")
    console.print()

    folder_uri = dataset_root.resolve().as_uri()
    console.print(f"[dim]Saved to[/] [cyan][link={folder_uri}]{dataset_root.resolve()}[/link][/]")
    console.print()

    _open_folder(dataset_root)


@click.group()
@click.option("--debug", is_flag=True, help="Enable verbose logging")
def cli(debug: bool) -> None:
    """prompt2dataset - build image datasets from a prompt."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.CRITICAL,
        format="%(levelname)s %(name)s: %(message)s",
    )


@cli.command()
def add() -> None:
    """Populate the current directory with images.

    On first run: prompts for a dataset description, resolves subjects, and
    downloads images. On subsequent runs: reuses the saved prompt, skips
    subjects already present, and fetches only new ones.
    """
    _ensure_api_key()
    dataset_root = Path.cwd()
    ds = _run_add(dataset_root)
    if ds:
        _print_summary(ds, dataset_root)


@cli.command()
@click.option("--misclassified", is_flag=True, help="Only review images misclassified by p2d train.")
def review(misclassified: bool) -> None:
    """Interactively review pending items in the current directory.

    V = valid, I = invalid, S = skip, Q = quit.
    """
    dataset_root = Path.cwd()
    ds = load_dataset(dataset_root)
    pending = ds.pending_review()

    if misclassified:
        mc_file = meta_dir(dataset_root) / "misclassified.json"
        if not mc_file.exists():
            raise click.ClickException("No misclassified.json found. Run `p2d train` first.")
        mc_paths: set[str] = {
            item["path"]
            for item in json.loads(mc_file.read_text(encoding="utf-8"))
        }
        pending = [i for i in pending if str((dataset_root / i.local_path).resolve()) in mc_paths]

    if not pending:
        console.print("[green]No items pending review.[/]")
        return

    console.print(f"\n[bold]{len(pending)} items to review[/] in [cyan]{dataset_root.name}[/]\n")

    reviewed = 0
    for idx, item in enumerate(pending, 1):
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("key", style="dim", no_wrap=True)
        table.add_column("value")
        table.add_row("item", item.item_id)
        table.add_row("label", item.label)
        table.add_row("url", item.source_url)
        img_path = dataset_root / item.local_path
        if img_path.exists():
            file_uri = img_path.resolve().as_uri()
            table.add_row("file", f"[cyan][link={file_uri}]{img_path}[/link][/]")
        for k, v in item.meta.items():
            if v:
                table.add_row(k, str(v))

        console.print(Panel(table, title=f"[bold]{idx}/{len(pending)}[/]", border_style="blue"))
        console.print("  [bold green]A[/] accept   [bold red]D[/] delete   [bold]S[/] skip   [bold]Q[/] quit\n")

        while True:
            key = click.prompt("  Action", default="s").strip().lower()
            if key in ("a", "accept"):
                item.review_status = ReviewStatus.valid
                reviewed += 1
                console.print("  [green]v accepted[/]\n")
                break
            elif key in ("d", "delete"):
                img_path = dataset_root / item.local_path
                if img_path.exists():
                    img_path.unlink()
                ds.items = [i for i in ds.items if i.item_id != item.item_id]
                mc_file = meta_dir(dataset_root) / "misclassified.json"
                if mc_file.exists():
                    mc = json.loads(mc_file.read_text(encoding="utf-8"))
                    mc = [e for e in mc if e["path"] != str(img_path.resolve())]
                    mc_file.write_text(json.dumps(mc, indent=2), encoding="utf-8")
                reviewed += 1
                console.print("  [red]x deleted[/]\n")
                break
            elif key in ("s", "skip", ""):
                console.print("  [dim]skipped[/]\n")
                break
            elif key in ("q", "quit"):
                save_dataset(ds, dataset_root)
                console.print(f"\n[bold]Saved.[/] Reviewed {reviewed} items.")
                return
            else:
                console.print("  [yellow]Unknown key. A, D, S, or Q[/]")

    save_dataset(ds, dataset_root)
    stats = ds.stats()
    console.print(Panel(
        f"[bold]Reviewed:[/] {reviewed}   "
        f"[bold]Valid:[/] {stats['valid']}   "
        f"[bold]Invalid:[/] {stats['invalid']}   "
        f"[bold]Pending:[/] {stats['pending']}",
        title="[green]Review complete[/]",
    ))


@cli.command()
def info() -> None:
    """Print a summary of the dataset in the current directory."""
    dataset_root = Path.cwd()
    ds = load_dataset(dataset_root)
    stats = ds.stats()

    table = Table(title=f"Dataset: {ds.dataset_id}", show_lines=True)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Prompt", ds.prompt)
    table.add_row("Sources", ", ".join(ds.sources))
    table.add_row("Subjects", str(len(ds.subjects)))
    table.add_row("Total images", str(stats["total"]))
    table.add_row("Pending review", str(stats["pending"]))
    table.add_row("Valid", str(stats["valid"]))
    table.add_row("Invalid", str(stats["invalid"]))
    console.print(table)

    if ds.subjects:
        console.print("\n[bold]Subjects:[/]")
        console.print(Columns([f"  [dim]{s}[/]" for s in ds.subjects], equal=True))


cli.add_command(train_cmd)


if __name__ == "__main__":
    cli()

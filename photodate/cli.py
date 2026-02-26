from datetime import date
from pathlib import Path

import openai
import typer
from rich.console import Console
from rich.table import Table

from .analyzer import analyze_album_full
from .exif import parse_exif_date, read_exif_date, write_exif_date
from .models import PhotoInfo
from .validator import scan_and_validate

app = typer.Typer(help="Photodate - Corrigeer datums van ingescande foto's")
console = Console()


def _load_photos(folder: Path) -> list[PhotoInfo]:
    """Load photos from a folder and read EXIF dates."""
    extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )
    if not files:
        console.print("[red]Geen foto's gevonden in deze map.[/red]")
        raise typer.Exit(1)

    photos = []
    for i, f in enumerate(files):
        photos.append(PhotoInfo(
            path=f,
            index=i,
            original_exif_date=read_exif_date(f),
        ))
    return photos


@app.command()
def validate(
    folder: Path = typer.Argument(..., help="Map met foto's om te valideren"),
):
    """Controleer foto's op datumfouten (toekomst, te oud, grote sprongen)."""
    if not folder.is_dir():
        console.print(f"[red]Map niet gevonden: {folder}[/red]")
        raise typer.Exit(1)

    photos = scan_and_validate(folder)

    table = Table(title=f"Validatie: {folder.name}")
    table.add_column("#", style="dim")
    table.add_column("Bestand")
    table.add_column("EXIF datum")
    table.add_column("Problemen")

    issues_found = 0
    for photo in photos:
        if not photo.issues:
            continue
        issues_found += len(photo.issues)
        issues_text = "\n".join(
            f"[{'red' if i.severity == 'error' else 'yellow'}]{i.description}[/]"
            for i in photo.issues
        )
        table.add_row(
            str(photo.index + 1),
            photo.path.name,
            photo.original_exif_date or "-",
            issues_text,
        )

    if issues_found == 0:
        console.print("[green]Geen problemen gevonden![/green]")
    else:
        console.print(table)
        console.print(f"\n[bold]{issues_found} probleem/problemen gevonden.[/bold]")


@app.command()
def analyze(
    folder: Path = typer.Argument(..., help="Map met ingescande foto's"),
    context: str = typer.Option(
        ..., "--context", "-c",
        help="Beschrijving van het album (bijv. 'Vakantie Spanje, zomer 1987')",
    ),
    no_backup: bool = typer.Option(
        False, "--no-backup", help="Maak geen backup bij het schrijven",
    ),
):
    """Analyseer foto's met AI en stel correcte datums voor."""
    if not folder.is_dir():
        console.print(f"[red]Map niet gevonden: {folder}[/red]")
        raise typer.Exit(1)

    photos = _load_photos(folder)
    console.print(f"[bold]{len(photos)} foto's gevonden in {folder.name}[/bold]\n")

    # Analyze with Claude
    console.print("Foto's analyseren met AI...\n")
    client = openai.OpenAI()
    photos = analyze_album(photos, context, client)

    # Show results
    table = Table(title=f"Datumschattingen: {folder.name}")
    table.add_column("#", style="dim")
    table.add_column("Bestand")
    table.add_column("Huidige datum", style="dim")
    table.add_column("Geschatte datum", style="bold green")
    table.add_column("Zekerheid")
    table.add_column("Redenering")

    for photo in photos:
        confidence_color = {
            "high": "green", "medium": "yellow", "low": "red",
        }.get(photo.estimate.confidence, "white") if photo.estimate else "white"

        table.add_row(
            str(photo.index + 1),
            photo.path.name,
            photo.original_exif_date or "-",
            str(photo.estimate.date) if photo.estimate else "-",
            f"[{confidence_color}]{photo.estimate.confidence}[/]"
            if photo.estimate else "-",
            photo.estimate.reasoning[:60] + "..."
            if photo.estimate and len(photo.estimate.reasoning) > 60
            else (photo.estimate.reasoning if photo.estimate else "-"),
        )

    console.print(table)

    # Ask for confirmation
    choice = typer.prompt(
        "\nWat wil je doen? [a]lles toepassen / [n]iets doen / [e]dit per foto",
        default="n",
    )

    if choice.lower() == "a":
        _apply_all(photos, backup=not no_backup)
    elif choice.lower() == "e":
        _edit_and_apply(photos, backup=not no_backup)
    else:
        console.print("Geen wijzigingen aangebracht.")


def _apply_all(photos: list[PhotoInfo], backup: bool = True) -> None:
    """Apply all estimated dates."""
    count = 0
    for photo in photos:
        if photo.estimate:
            write_exif_date(photo.path, photo.estimate.date, backup=backup)
            count += 1
    console.print(f"[green]{count} foto's bijgewerkt.[/green]")


def _edit_and_apply(photos: list[PhotoInfo], backup: bool = True) -> None:
    """Let user edit individual dates before applying."""
    for photo in photos:
        if not photo.estimate:
            continue
        console.print(
            f"\n[bold]{photo.path.name}[/bold]: "
            f"geschat {photo.estimate.date} ({photo.estimate.confidence})"
        )
        console.print(f"  Reden: {photo.estimate.reasoning}")

        action = typer.prompt(
            "  [a]ccepteren / [s]kip / datum invoeren (YYYY-MM-DD)",
            default="a",
        )

        if action.lower() == "a":
            write_exif_date(photo.path, photo.estimate.date, backup=backup)
            console.print("  [green]✓ Toegepast[/green]")
        elif action.lower() == "s":
            console.print("  [dim]Overgeslagen[/dim]")
        else:
            try:
                custom_date = date.fromisoformat(action)
                write_exif_date(photo.path, custom_date, backup=backup)
                console.print(f"  [green]✓ Toegepast: {custom_date}[/green]")
            except ValueError:
                console.print("  [red]Ongeldige datum, overgeslagen.[/red]")


if __name__ == "__main__":
    app()

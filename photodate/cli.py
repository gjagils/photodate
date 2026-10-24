from datetime import date
from pathlib import Path
from typing import Optional

import openai
import typer
from rich.console import Console
from rich.table import Table

from . import collection as coll
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


def _default_backup_root() -> Path:
    from .storage import STORAGE_DIR
    return STORAGE_DIR / "backups"


def _album_matches(album: coll.CollectionAlbum, wanted: list[str]) -> bool:
    if not wanted:
        return True
    for w in wanted:
        order = coll.roman_to_int(w)
        if order is not None and album.order == order:
            return True
        if order is None and w.lower() in album.name.lower():
            return True
    return False


@app.command()
def collection(
    folder: Path = typer.Argument(..., help="Map met albums, bijv. 'Fotoboeken van Gils'"),
    album: list[str] = typer.Option(
        [], "--album", "-a", help="Alleen dit album: Romeins cijfer (IX) of deel van de naam",
    ),
    apply: bool = typer.Option(False, "--apply", help="Schrijf de datums echt weg"),
    backup_root: Optional[Path] = typer.Option(
        None, "--backup-dir", help="Map voor originelen (standaard DATA_DIR/backups)",
    ),
):
    """Dateer een hele collectie albums op basis van mapnaam en volgorde, zonder AI."""
    if not folder.is_dir():
        console.print(f"[red]Map niet gevonden: {folder}[/red]")
        raise typer.Exit(1)

    backup_root = backup_root or _default_backup_root()
    albums = [a for a in coll.load_collection(folder, folder.name) if _album_matches(a, album)]

    table = Table(title=f"Collectie: {folder.name}")
    table.add_column("Album")
    table.add_column("Foto's", justify="right")
    table.add_column("Periode")
    table.add_column("Bron", style="dim")
    table.add_column("Eerste / midden / laatste")
    for a in albums:
        plan = a.planned()
        sample = (
            " / ".join(f"{p.name} {dt:%Y-%m-%d}" for p, dt in (plan[0], plan[len(plan) // 2], plan[-1]))
            if plan else "[yellow]periode onvolledig[/yellow]"
        )
        table.add_row(
            a.name, str(len(a.photos)), coll.period_label(a.start, a.end),
            f"{a.start_source or '-'} / {a.end_source or '-'}", sample,
        )
    console.print(table)

    if not apply:
        console.print("\n[dim]Proefweergave. Gebruik --apply om te schrijven.[/dim]")
        return

    for a in albums:
        if not a.complete:
            console.print(f"[yellow]Overgeslagen (periode onvolledig): {a.name}[/yellow]")
            continue
        written, errors = coll.apply_album(a, coll.backup_dir_for(backup_root, a.rel))
        console.print(f"[green]{a.name}: {len(written)} foto's gedateerd[/green]"
                      + (f", [red]{len(errors)} fouten[/red]" if errors else ""))
        for e in errors[:5]:
            console.print(f"  [red]{e}[/red]")


@app.command()
def restore(
    album_folder: Path = typer.Argument(..., help="Albummap om terug te zetten"),
    backup_root: Optional[Path] = typer.Option(
        None, "--backup-dir", help="Map met originelen (standaard DATA_DIR/backups)",
    ),
):
    """Zet de originelen van een album terug (na 'collection --apply')."""
    backup_root = backup_root or _default_backup_root()
    rel = f"{album_folder.parent.name}/{album_folder.name}"
    n = coll.restore_album(album_folder, coll.backup_dir_for(backup_root, rel))
    console.print(f"[green]{n} originelen teruggezet.[/green]")


@app.command("fill-dates")
def fill_dates(
    folder: Path = typer.Argument(..., help="Map uit de fotobibliotheek, bijv. Fotobibliotheek/2019"),
    apply: bool = typer.Option(False, "--apply", help="Schrijf de datums echt weg (JPEG; HEIC/PNG/TIFF met exiftool)"),
    backup_root: Optional[Path] = typer.Option(None, "--backup-dir", help="Map voor originelen"),
    report: Optional[Path] = typer.Option(None, "--csv", help="Schrijf het plan als CSV"),
    year_only: bool = typer.Option(False, "--ook-alleen-jaar", help="Schrijf ook foto's waarvan alleen het jaar bekend is"),
):
    """Herstel ontbrekende opnamedatums uit Takeout-JSON, bestandsnaam, origineel, bestandsdatum of map."""
    import collections
    import csv

    from . import filldates as fd

    if not folder.is_dir():
        console.print(f"[red]Map niet gevonden: {folder}[/red]")
        raise typer.Exit(1)
    plans = fd.plan_folder(folder)

    table = Table(title=f"Zonder datum: {folder} ({len(plans)} foto's)")
    for col in ("Submap", "Totaal", "Takeout", "Bestandsnaam", "Origineel", "Bestandsdatum", "Map", "Alleen jaar", "Niet schrijfbaar", "Andere maand"):
        table.add_column(col, justify="right" if col != "Submap" else "left")
    by_dir = collections.defaultdict(list)
    for p in plans:
        by_dir[p.path.parent.relative_to(folder) if p.path.parent != folder else Path(".")].append(p)
    for sub, items in sorted(by_dir.items(), key=lambda kv: str(kv[0])):
        src = collections.Counter(p.source for p in items)
        year, month = fd.folder_month(items[0].path)
        other = sum(1 for p in items if p.when and month and (p.when.year, p.when.month) != (year, month))
        non_jpeg = sum(1 for p in items if not fd.can_write(p.path))
        table.add_row(str(sub), str(len(items)), str(src["takeout"]), str(src["bestandsnaam"]), str(src["origineel"]),
                      str(src["bestandsdatum"]), str(src["map"]), str(src["jaar"] + src["geen"]), str(non_jpeg), str(other))
    console.print(table)

    if report:
        with report.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pad", "bron", "datum", "toelichting"])
            for p in plans:
                w.writerow([str(p.path.relative_to(folder)), p.source, p.when.isoformat() if p.when else "", p.note])
        console.print(f"Plan geschreven naar {report}")

    if not apply:
        console.print("\n[dim]Proefweergave. Gebruik --apply om JPEG-bestanden te schrijven.[/dim]")
        return
    backup_root = (backup_root or _default_backup_root()) / str(folder).strip("/").replace("/", "__")
    written = errors = skipped = 0
    done = []
    for p in plans:
        if not p.when or not fd.can_write(p.path) or (p.source == "jaar" and not year_only):
            skipped += 1
            continue
        try:
            fd.write_plan(p, backup_root)
            written += 1
            done.append(p)
        except Exception as exc:
            errors += 1
            console.print(f"[red]{p.path.name}: {exc}[/red]")
    if done:
        console.print(f"Bestandsdatums nagelopen: {fd.settle_mtimes(done)} opnieuw gezet")
    console.print(f"[green]{written} geschreven[/green], {skipped} overgeslagen (niet schrijfbaar, alleen jaar of geen datum), {errors} fouten")


if __name__ == "__main__":
    app()

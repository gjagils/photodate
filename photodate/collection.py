"""Date whole collections of scanned albums from folder names and photo order, without AI."""
import calendar
import io
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import piexif

logger = logging.getLogger(__name__)

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".heic", ".heif"}
SKIP_DIRS = {"@eaDir", "#recycle", ".git", "nietzelfgenomen"}

MONTHS = {
    "jan": 1, "januari": 1,
    "feb": 2, "februari": 2,
    "mrt": 3, "maart": 3, "mar": 3,
    "apr": 4, "april": 4,
    "mei": 5,
    "jun": 6, "juni": 6,
    "jul": 7, "juli": 7,
    "aug": 8, "augustus": 8,
    "sep": 9, "sept": 9, "september": 9,
    "okt": 10, "oktober": 10, "oct": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
MONTH_NAMES = [
    "", "januari", "februari", "maart", "april", "mei", "juni",
    "juli", "augustus", "september", "oktober", "november", "december",
]

ROMAN_RE = re.compile(r"^\s*([IVXLC]+)(?=\s|$|-)")
# Full dates first; the month-year branch refuses digits followed by "-" so "21-9-74" stays whole.
TOKEN_RE = re.compile(
    r"(?P<d>\d{1,2})-(?P<m>\d{1,2})-(?P<y>\d{2,4})"
    r"|(?P<mon>[a-z]+)\.?\s*['’`;]?\s*(?P<my>\d{2,4})(?![\d-])"
)
SEQ_RE = re.compile(r"_(\d+)$")
CAMERA_PREFIX_RE = re.compile(r"^(img|dsc|dscn|dscf|pxl|p)_?\d+$", re.IGNORECASE)


def _year(raw: str) -> int:
    y = int(raw)
    if len(raw) <= 2:
        y += 1900 if y >= 30 else 2000
    return y


def _month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def roman_to_int(s: str) -> Optional[int]:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    s = s.upper()
    if not s or any(c not in values for c in s):
        return None
    total = 0
    for i, c in enumerate(s):
        v = values[c]
        if i + 1 < len(s) and values[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


def album_order(name: str) -> Optional[int]:
    m = ROMAN_RE.match(name)
    return roman_to_int(m.group(1)) if m else None


def parse_period(name: str) -> tuple[Optional[date], Optional[date]]:
    """Read an album period like "Okt '88 tot Jan '90" or "Vanaf 21-9-74" from a folder name."""
    text = ROMAN_RE.sub("", name, count=1).lower()
    tokens: list[tuple[int, date, date]] = []
    for m in TOKEN_RE.finditer(text):
        if m.group("d"):
            try:
                d = date(_year(m.group("y")), int(m.group("m")), int(m.group("d")))
            except ValueError:
                continue
            tokens.append((m.start(), d, d))
        else:
            month = MONTHS.get(m.group("mon"))
            if not month:
                continue
            y = _year(m.group("my"))
            tokens.append((m.start(), date(y, month, 1), _month_end(y, month)))

    if not tokens:
        return None, None
    if len(tokens) >= 2:
        return tokens[0][1], tokens[-1][2]

    pos, first, last = tokens[0]
    before = text[:pos]
    if re.search(r"\bvanaf\b", before):
        return first, None
    if re.search(r"\btot\b", before):
        return None, last
    return first, last


def _seq_number(path: Path) -> Optional[int]:
    stem = path.stem
    if CAMERA_PREFIX_RE.match(stem):
        return None
    m = SEQ_RE.search(stem)
    return int(m.group(1)) if m else None


def order_photos(paths: list[Path]) -> list[Path]:
    """Album order: scan sequence number; a few unnumbered files are cover shots and go first."""
    numbered = sorted(
        (p for p in paths if _seq_number(p) is not None),
        key=lambda p: (_seq_number(p), p.name.lower()),
    )
    unnumbered = sorted((p for p in paths if _seq_number(p) is None), key=lambda p: p.name.lower())
    if len(unnumbered) <= 2:
        return unnumbered + numbered
    return numbered + unnumbered


def spread_datetimes(n: int, start: date, end: date) -> list[datetime]:
    """Spread n photos evenly over [start, end], keeping order unique (12:00, 12:01, ... on shared days)."""
    if n <= 0:
        return []
    span = max((end - start).days, 0)
    result = []
    prev_day = None
    minute = 0
    for i in range(n):
        offset = round(i * span / (n - 1)) if n > 1 else 0
        day = start + timedelta(days=offset)
        minute = minute + 1 if day == prev_day else 0
        result.append(datetime(day.year, day.month, day.day, 12, 0) + timedelta(minutes=minute))
        prev_day = day
    return result


def period_label(start: Optional[date], end: Optional[date]) -> str:
    def fmt(d: Optional[date]) -> str:
        return f"{MONTH_NAMES[d.month][:3]} {d.year}" if d else "?"
    return f"{fmt(start)} – {fmt(end)}"


def describe_estimate(dt: datetime, album_name: str, start: date, end: date) -> str:
    # EXIF ImageDescription is ASCII-only, so no en dash here.
    period = period_label(start, end).replace("–", "-")
    return f"Datum geschat: {MONTH_NAMES[dt.month]} {dt.year} (album {album_name}, {period})"


@dataclass
class CollectionAlbum:
    folder: Path
    rel: str
    name: str
    order: Optional[int]
    photos: list[Path] = field(default_factory=list)
    start: Optional[date] = None
    end: Optional[date] = None
    start_source: str = ""   # "naam" | "buren" | "handmatig" | ""
    end_source: str = ""

    @property
    def complete(self) -> bool:
        return self.start is not None and self.end is not None and self.start <= self.end

    def planned(self) -> list[tuple[Path, datetime]]:
        if not self.complete:
            return []
        return list(zip(self.photos, spread_datetimes(len(self.photos), self.start, self.end)))


def parse_manual_date(value: str, is_end: bool) -> Optional[date]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        if len(value) == 7:
            y, m = int(value[:4]), int(value[5:7])
            return _month_end(y, m) if is_end else date(y, m, 1)
        return date.fromisoformat(value)
    except ValueError:
        return None


def list_photos(folder: Path) -> list[Path]:
    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in PHOTO_EXTENSIONS and not f.name.startswith(".")
    ]


def load_collection(
    folder: Path,
    rel_prefix: str = "",
    manual: Optional[dict[str, tuple[str, str]]] = None,
) -> list[CollectionAlbum]:
    """Albums directly under `folder`, with periods from manual input, folder name, then neighbours."""
    manual = manual or {}
    albums = []
    for sub in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not sub.is_dir() or sub.name in SKIP_DIRS or sub.name.startswith((".", "_")):
            continue
        photos = list_photos(sub)
        if not photos:
            continue
        rel = f"{rel_prefix}/{sub.name}" if rel_prefix else sub.name
        album = CollectionAlbum(
            folder=sub, rel=rel, name=sub.name, order=album_order(sub.name),
            photos=order_photos(photos),
        )
        start, end = parse_period(sub.name)
        if start:
            album.start, album.start_source = start, "naam"
        if end:
            album.end, album.end_source = end, "naam"
        m_start, m_end = manual.get(rel, ("", ""))
        if (d := parse_manual_date(m_start, is_end=False)):
            album.start, album.start_source = d, "handmatig"
        if (d := parse_manual_date(m_end, is_end=True)):
            album.end, album.end_source = d, "handmatig"
        albums.append(album)

    ordered = sorted((a for a in albums if a.order is not None), key=lambda a: a.order)
    for i, album in enumerate(ordered):
        prev = ordered[i - 1] if i > 0 else None
        nxt = ordered[i + 1] if i + 1 < len(ordered) else None
        if album.start is None and prev and prev.end and prev.end_source != "buren":
            album.start, album.start_source = prev.end, "buren"
        if album.end is None and nxt and nxt.start and nxt.start_source != "buren":
            album.end, album.end_source = nxt.start, "buren"

    albums.sort(key=lambda a: (a.order is None, a.order or 0, a.name.lower()))
    return albums


# ── Writing ───────────────────────────────────────────────────────────────────

def _minimal_exif(exif: dict) -> dict:
    keep = {piexif.ImageIFD.Orientation}
    return {
        "0th": {k: v for k, v in exif.get("0th", {}).items() if k in keep},
        "Exif": {}, "1st": {}, "GPS": {},
    }


def write_estimated_date(path: Path, dt: datetime, description: str, backup_dir: Path) -> None:
    """Write the estimated date + description into EXIF (no re-encode); keep the first original."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / path.name
    if not backup.exists():
        shutil.copy2(path, backup)

    stamp = dt.strftime("%Y:%m:%d %H:%M:%S").encode()
    try:
        exif = piexif.load(str(path))
    except Exception:
        exif = {"0th": {}, "Exif": {}, "1st": {}, "GPS": {}}
    exif.pop("thumbnail", None)
    exif["1st"] = {}

    def apply(e: dict) -> bytes:
        e["Exif"][piexif.ExifIFD.DateTimeOriginal] = stamp
        e["Exif"][piexif.ExifIFD.DateTimeDigitized] = stamp
        e["0th"][piexif.ImageIFD.DateTime] = stamp
        e["0th"][piexif.ImageIFD.ImageDescription] = description.encode("ascii", "replace")
        return piexif.dump(e)

    try:
        exif_bytes = apply(exif)
    except Exception:
        # Phone EXIF sometimes holds tags piexif can't re-serialise; keep only what matters.
        exif_bytes = apply(_minimal_exif(exif))
    out = io.BytesIO()
    piexif.insert(exif_bytes, path.read_bytes(), out)
    # On SMB/NFS the server stamps mtime when buffered writes land; flush before setting it.
    with open(path, "r+b") as fh:
        fh.write(out.getvalue())
        fh.truncate()
        fh.flush()
        os.fsync(fh.fileno())

    try:
        ts = time.mktime(dt.timetuple())
        os.utime(path, (ts, ts))
    except (OSError, OverflowError) as exc:
        logger.info("Kon bestandsdatum niet zetten voor %s: %s", path.name, exc)


def apply_album(album: CollectionAlbum, backup_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Write spread dates for one album. Returns ({filename: iso datetime}, errors)."""
    written: dict[str, str] = {}
    errors: list[str] = []
    for path, dt in album.planned():
        try:
            write_estimated_date(
                path, dt, describe_estimate(dt, album.name, album.start, album.end), backup_dir
            )
            written[path.name] = dt.isoformat(timespec="minutes")
        except Exception as exc:
            logger.warning("Kon %s niet dateren: %s", path, exc)
            errors.append(f"{path.name}: {exc}")
    return written, errors


def backup_dir_for(backup_root: Path, album_rel: str) -> Path:
    return backup_root / album_rel.replace("/", "__").replace("\\", "__")


def restore_album(folder: Path, backup_dir: Path) -> int:
    if not backup_dir.is_dir():
        return 0
    restored = 0
    for backup in backup_dir.iterdir():
        target = folder / backup.name
        if backup.is_file() and target.exists():
            shutil.copy2(backup, target)
            restored += 1
    return restored

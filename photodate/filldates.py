"""Recover capture dates for library photos without EXIF date (Takeout JSON, file name, mtime, folder)."""
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import piexif
from PIL import Image

from .collection import MONTH_NAMES, rewrite_exif

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

logger = logging.getLogger(__name__)

LOCAL_TZ = ZoneInfo("Europe/Amsterdam")
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".gif"}
WRITABLE = {".jpg", ".jpeg"}  # EXIF in place without re-encoding
SKIP_DIRS = {"@eaDir", "#recycle", ".git", "nietzelfgenomen"}
FIRST_YEAR = 1990

NAME_DATE_RE = re.compile(
    r"(?<!\d)(19[89]\d|20[0-4]\d)[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])"
    r"(?:[-_ .T]?([01]\d|2[0-3])[-_.:]?([0-5]\d)[-_.:]?([0-5]\d))?(?!\d)"
)
COPY_SUFFIX_RE = re.compile(r"^(.*?)(\(\d+\))$")
EDITED_RE = re.compile(r"-(bewerkt|edited)$", re.IGNORECASE)


@dataclass
class Plan:
    path: Path
    when: Optional[datetime]
    source: str          # takeout | bestandsnaam | origineel | bestandsdatum | map | jaar | geen
    note: str = ""


def exif_date(path: Path) -> Optional[str]:
    try:
        if path.suffix.lower() in WRITABLE:
            exif = piexif.load(str(path))
        else:
            raw = Image.open(path).info.get("exif")
            if not raw:
                return None
            exif = piexif.load(raw)
    except Exception:
        return None
    for ifd, tag in (("Exif", piexif.ExifIFD.DateTimeOriginal), ("Exif", piexif.ExifIFD.DateTimeDigitized),
                     ("0th", piexif.ImageIFD.DateTime)):
        value = (exif.get(ifd, {}).get(tag) or b"").decode(errors="ignore").strip()
        if re.match(r"^\d{4}:\d{2}:\d{2}", value) and not value.startswith("0000"):
            return value
    return None


def folder_month(path: Path) -> tuple[Optional[int], Optional[int]]:
    """(year, month) from a .../YYYY/MM/ or .../YYYY/ folder; month 00 means unknown."""
    parent, grand = path.parent.name, path.parent.parent.name
    if re.fullmatch(r"\d{2}", parent) and re.fullmatch(r"(19|20)\d{2}", grand):
        month = int(parent)
        return int(grand), (month if 1 <= month <= 12 else None)
    if re.fullmatch(r"(19|20)\d{2}", parent):
        return int(parent), None
    return None, None


def sidecar_for(path: Path, jsons: dict[str, Path]) -> Optional[Path]:
    """Find the Google Takeout JSON next to a photo (handles (1)-copies, -bewerkt and truncation)."""
    name = path.name
    stem, ext = os.path.splitext(name)
    candidates = [f"{name}.json", f"{name}.supplemental-metadata.json"]
    if (m := COPY_SUFFIX_RE.match(stem)):
        candidates += [f"{m.group(1)}{ext}{m.group(2)}.json", f"{m.group(1)}{ext}.json"]
    if (m := EDITED_RE.search(stem)):
        base = stem[: m.start()]
        candidates += [f"{base}{ext}.json", f"{base}{ext}.supplemental-metadata.json"]
    for c in candidates:
        if c.lower() in jsons:
            return jsons[c.lower()]
    # Takeout truncates long names; accept a JSON whose base is a prefix of ours.
    long_base = f"{name}.supplemental-metadata".lower()
    for key, p in jsons.items():
        base = key[:-5]
        if len(base) >= 30 and long_base.startswith(base):
            return p
    return None


def takeout_time(sidecar: Path) -> Optional[datetime]:
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        ts = int(data["photoTakenTime"]["timestamp"])
    except Exception:
        return None
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).astimezone(LOCAL_TZ).replace(tzinfo=None)


def name_time(name: str) -> Optional[datetime]:
    for m in NAME_DATE_RE.finditer(name):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        h, mi, s = (int(x) for x in (m.group(4) or 12, m.group(5) or 0, m.group(6) or 0))
        try:
            dt = datetime(y, mo, d, h, mi, s)
        except ValueError:
            continue
        if FIRST_YEAR <= y and dt <= datetime.now():
            return dt
    return None


def original_name(name: str) -> Optional[str]:
    """'IMG_1(1).HEIC' / 'x-bewerkt.jpg' -> the name of the photo it was copied or edited from."""
    stem, ext = os.path.splitext(name)
    base = EDITED_RE.sub("", COPY_SUFFIX_RE.sub(r"\1", stem))
    return f"{base}{ext}" if base != stem else None


def same_picture(copy: Path, original: Path) -> bool:
    """A copy may be resized, but keeps the aspect ratio (cover(12).jpg vs cover.jpg usually doesn't)."""
    try:
        (w1, h1), (w2, h2) = Image.open(copy).size, Image.open(original).size
    except Exception:
        return False
    if not (w1 and h1 and w2 and h2):
        return False
    r1, r2 = w1 / h1, w2 / h2
    return abs(r1 - r2) / max(r1, r2) <= 0.02 or abs(1 / r1 - r2) / max(1 / r1, r2) <= 0.02


def plan_photo(path: Path, jsons: dict[str, Path],
               originals: Optional[dict[str, tuple[datetime, Path]]] = None) -> Plan:
    year, month = folder_month(path)
    side = sidecar_for(path, jsons)
    if side and (dt := takeout_time(side)):
        return Plan(path, dt, "takeout")
    if (dt := name_time(path.name)):
        return Plan(path, dt, "bestandsnaam")
    orig = original_name(path.name)
    if orig and originals and (hit := originals.get(orig.lower())):
        dt, orig_path = hit
        if EDITED_RE.search(os.path.splitext(path.name)[0]) or same_picture(path, orig_path):
            return Plan(path, dt, "origineel", orig)
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    if year and month and (mtime.year, mtime.month) == (year, month):
        return Plan(path, mtime, "bestandsdatum")
    if year and month:
        return Plan(path, datetime(year, month, 15, 12, 0), "map", f"{MONTH_NAMES[month]} {year}")
    if year:
        # Only the year is known; written only on request (see --ook-alleen-jaar).
        return Plan(path, datetime(year, 7, 1, 12, 0), "jaar", f"{year}, maand onbekend")
    return Plan(path, None, "geen")


def scan_dates(folder: Path, workers: int = 16) -> dict[Path, Optional[str]]:
    paths = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(("_", "."))]
        for f in filenames:
            if not f.startswith(".") and Path(f).suffix.lower() in PHOTO_EXTENSIONS:
                paths.append(Path(dirpath) / f)
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip(paths, pool.map(exif_date, paths)))


def plan_folder(folder: Path, workers: int = 16) -> list[Plan]:
    dates = scan_dates(folder, workers)
    missing = sorted(p for p, d in dates.items() if d is None)
    originals: dict[str, tuple[datetime, Path]] = {}
    for p, d in dates.items():
        if d:
            try:
                originals.setdefault(p.name.lower(), (datetime.strptime(d[:19], "%Y:%m:%d %H:%M:%S"), p))
            except ValueError:
                pass
    jsons_by_dir: dict[Path, dict[str, Path]] = {}

    def jsons(d: Path) -> dict[str, Path]:
        if d not in jsons_by_dir:
            jsons_by_dir[d] = {p.name.lower(): p for p in d.iterdir() if p.suffix.lower() == ".json"}
        return jsons_by_dir[d]

    # Originals first, so copies and edits can borrow a recovered date too.
    ordered = sorted(missing, key=lambda p: original_name(p.name) is not None)
    plans = []
    for path in ordered:
        plan = plan_photo(path, jsons(path.parent), originals)
        if plan.source in ("takeout", "bestandsnaam"):
            originals.setdefault(path.name.lower(), (plan.when, path))
        plans.append(plan)
    _make_unique(plans)
    return sorted(plans, key=lambda p: str(p.path))


def _make_unique(plans: list[Plan]) -> None:
    """Folder-based guesses share a day; give them distinct minutes in name order."""
    seen: dict[tuple, int] = {}
    for p in sorted((p for p in plans if p.source in ("map", "jaar")), key=lambda p: p.path.name.lower()):
        key = (p.path.parent, p.when)
        n = seen.get(key, 0)
        seen[key] = n + 1
        p.when = p.when + timedelta(minutes=n)


def description(plan: Plan) -> str:
    if plan.source == "takeout":
        return "Datum uit Google Takeout"
    if plan.source == "bestandsnaam":
        return "Datum uit bestandsnaam"
    if plan.source == "bestandsdatum":
        return "Datum uit bestandsdatum"
    if plan.source == "origineel":
        return f"Datum van origineel {plan.note}"
    return f"Datum geschat: {plan.note} (uit map)"


def settle_mtimes(plans: list[Plan], wait: float = 15.0) -> int:
    """On SMB the NAS can stamp mtime once delayed closes land; set it again afterwards."""
    time.sleep(wait)
    fixed = 0
    for p in plans:
        ts = time.mktime(p.when.timetuple())
        try:
            if abs(p.path.stat().st_mtime - ts) > 1:
                os.utime(p.path, (ts, ts))
                fixed += 1
        except OSError:
            pass
    return fixed


def write_plan(plan: Plan, backup_dir: Path) -> None:
    stamp = plan.when.strftime("%Y:%m:%d %H:%M:%S").encode()
    text = description(plan).encode("ascii", "replace")

    def mutate(e: dict) -> None:
        e["Exif"][piexif.ExifIFD.DateTimeOriginal] = stamp
        e["Exif"][piexif.ExifIFD.DateTimeDigitized] = stamp
        e["0th"][piexif.ImageIFD.DateTime] = stamp
        e["0th"].setdefault(piexif.ImageIFD.ImageDescription, text)

    rewrite_exif(plan.path, mutate, backup_dir / plan.path.parent.name, time.mktime(plan.when.timetuple()))

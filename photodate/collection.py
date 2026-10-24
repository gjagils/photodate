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
    return _to_datetimes(start, [round(i * span / (n - 1)) if n > 1 else 0 for i in range(n)])


def _to_datetimes(start: date, day_offsets: list[int]) -> list[datetime]:
    result = []
    prev_day = None
    minute = 0
    for offset in day_offsets:
        day = start + timedelta(days=offset)
        minute = minute + 1 if day == prev_day else 0
        result.append(datetime(day.year, day.month, day.day, 12, 0) + timedelta(minutes=minute))
        prev_day = day
    return result


REEKS_GAP_DAYS = 1.0


def spread_by_links(links: list[str], start: date, end: date) -> list[datetime]:
    """Same moment -> same day, same series -> next day, the rest of the period between new events."""
    n = len(links)
    if n == 0:
        return []
    span = max((end - start).days, 0)
    transitions = links[1:]
    reeks = transitions.count("reeks")
    nieuw = sum(1 for t in transitions if t not in ("moment", "reeks"))
    if nieuw:
        reeks_gap = min(REEKS_GAP_DAYS, span / (reeks + nieuw))
        nieuw_gap = (span - reeks * reeks_gap) / nieuw
    else:
        reeks_gap = span / reeks if reeks else 0.0
        nieuw_gap = 0.0
    gaps = {"moment": 0.0, "reeks": reeks_gap}

    pos = 0.0
    offsets = [0]
    for t in transitions:
        pos += gaps.get(t, nieuw_gap)
        offsets.append(min(round(pos), span))
    return _to_datetimes(start, offsets)


def moment_groups(links: list[str]) -> list[list[int]]:
    groups: list[list[int]] = []
    for i, link in enumerate(links):
        if i == 0 or link != "moment":
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


# Cost per month for a photo that looks like a given season (0 = fits, higher = unlikely).
SEASON_MONTH_COST = {
    "zomer": ({6: 0, 7: 0, 8: 0, 5: 1, 9: 1, 4: 3, 10: 3}, 6),
    "winter": ({12: 0, 1: 0, 2: 0, 11: 1, 3: 1, 10: 3, 4: 3}, 6),
    "herfst": ({9: 0, 10: 0, 11: 0, 8: 1, 12: 1, 7: 3, 1: 3}, 5),
    "lente": ({3: 0, 4: 0, 5: 0, 2: 1, 6: 1, 1: 3, 7: 3}, 5),
}
SEASON_WEIGHT = 8.0
DRIFT_PER_DAY = 0.02       # moving a moment away from the even spread
OUTSIDE_PER_DAY = 0.04     # extra, per day outside the album period
REEKS_EXTRA_DAY = 0.3      # per extra day between moments of one series
SLACK_DAYS = 120


def _moment_season(seasons: list[str], group: list[int]) -> tuple[str, int]:
    votes: dict[str, int] = {}
    for i in group:
        if seasons[i] in SEASON_MONTH_COST:
            votes[seasons[i]] = votes.get(seasons[i], 0) + 1
    if not votes:
        return "onbekend", 0
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return "onbekend", 0
    return ranked[0]


HOLIDAY_LABELS = {
    "sinterklaas": "Sinterklaas", "kerst": "kerst", "oudjaar": "oud en nieuw",
    "driekoningen": "Driekoningen", "carnaval": "carnaval", "pasen": "Pasen",
    "koninginnedag": "Koninginnedag", "sintmaarten": "Sint-Maarten", "verjaardag": "verjaardag",
}
HOLIDAY_PER_DAY = 0.5
# Low cap on purpose: a misrecognised holiday whose date the album order can't reach then
# costs the same everywhere, instead of dragging a whole run of moments along.
HOLIDAY_MAX = 10.0
HOLIDAY_MAX_WITH_AGE = 20.0  # a visible age matching a family birthday is stronger evidence


@dataclass
class Person:
    name: str
    born: date
    died: Optional[date] = None


def easter_sunday(year: int) -> date:
    """Gregorian Easter (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def holiday_windows(
    kind: str, first: date, last: date, family: list[Person] = (), age: Optional[int] = None,
) -> list[tuple[date, date]]:
    """Date ranges in [first, last] where a photo of this holiday would be taken."""
    windows: list[tuple[date, date]] = []
    for y in range(first.year, last.year + 1):
        if kind == "sinterklaas":
            windows.append((date(y, 11, 15), date(y, 12, 6)))
        elif kind == "kerst":
            windows.append((date(y, 12, 24), date(y, 12, 26)))
        elif kind == "oudjaar":
            windows.append((date(y, 12, 31), date(y + 1, 1, 1)))
        elif kind == "driekoningen":
            windows.append((date(y, 1, 5), date(y, 1, 7)))
        elif kind == "carnaval":
            ash_wednesday = easter_sunday(y) - timedelta(days=46)
            windows.append((ash_wednesday - timedelta(days=4), ash_wednesday - timedelta(days=1)))
        elif kind == "sintmaarten":
            windows.append((date(y, 11, 10), date(y, 11, 12)))
        elif kind == "pasen":
            e = easter_sunday(y)
            windows.append((e - timedelta(days=1), e + timedelta(days=1)))
        elif kind == "koninginnedag":
            d = date(y, 4, 30) if y < 2014 else date(y, 4, 27)
            windows.append((d, d))
        elif kind == "verjaardag":
            for p in family:
                if age is not None and y != p.born.year + age:
                    continue
                try:
                    birthday = date(y, p.born.month, p.born.day)
                except ValueError:  # 29 February
                    birthday = date(y, 2, 28)
                if birthday <= p.born or (p.died and birthday > p.died):
                    continue
                # Parties are often held on a weekend shortly before or after.
                windows.append((birthday - timedelta(days=3), birthday + timedelta(days=7)))
    return [(a, b) for a, b in windows if b >= first and a <= last]


def _moment_holiday(holidays: list[str], ages: list[Optional[int]], group: list[int]
                    ) -> tuple[str, Optional[int]]:
    votes: dict[str, int] = {}
    for i in group:
        if holidays[i] in HOLIDAY_LABELS:
            votes[holidays[i]] = votes.get(holidays[i], 0) + 1
    if not votes:
        return "geen", None
    kind = max(votes, key=votes.get)
    age = next((ages[i] for i in group if holidays[i] == "verjaardag" and ages[i]), None)
    return kind, age if kind == "verjaardag" else None


def place_with_seasons(
    links: list[str], seasons: list[str], start: date, end: date, slack_days: int = SLACK_DAYS,
    holidays: Optional[list[str]] = None, ages: Optional[list[Optional[int]]] = None,
    family: list[Person] = (),
) -> list[datetime]:
    """Order-preserving placement of moments that fits seasons and holidays, near the even spread.

    Dynamic programming over (moment, day): each moment gets one day, days strictly increase,
    and the cost adds season mismatch, distance to a recognised holiday or family birthday,
    drift from the even spread, days outside the album period and gaps inside a series.
    """
    holidays = holidays or ["geen"] * len(links)
    ages = ages or [None] * len(links)
    n = len(links)
    if n == 0:
        return []
    groups = moment_groups(links)
    lo = start - timedelta(days=slack_days)
    hi = end + timedelta(days=slack_days)
    total_days = (hi - lo).days + 1
    if len(groups) > total_days:
        return spread_by_links(links, start, end)

    base = spread_by_links(links, start, end)
    base_t = [(base[g[0]].date() - lo).days for g in groups]
    months = [(lo + timedelta(days=t)).month for t in range(total_days)]
    start_t, end_t = (start - lo).days, (end - lo).days
    inf = float("inf")

    def node_costs(m: int) -> list[float]:
        season, votes = _moment_season(seasons, groups[m])
        table, default = SEASON_MONTH_COST.get(season, ({}, 0))
        weight = SEASON_WEIGHT * min(1 + 0.25 * (votes - 1), 2.0) if votes else 0.0
        kind, age = _moment_holiday(holidays, ages, groups[m])
        windows = [
            ((a - lo).days, (b - lo).days)
            for a, b in holiday_windows(kind, lo, hi, family, age)
        ] if kind != "geen" else []
        cap = HOLIDAY_MAX_WITH_AGE if age else HOLIDAY_MAX
        costs = []
        for t in range(total_days):
            outside = max(0, start_t - t) + max(0, t - end_t)
            c = DRIFT_PER_DAY * abs(t - base_t[m]) + OUTSIDE_PER_DAY * outside
            if weight:
                c += weight * table.get(months[t], default)
            if windows:
                gap = min(max(a - t, 0, t - b) for a, b in windows)
                c += min(cap, HOLIDAY_PER_DAY * gap)
            costs.append(c)
        return costs

    prev = node_costs(0)
    back: list[list[int]] = []
    for m in range(1, len(groups)):
        node = node_costs(m)
        series = links[groups[m][0]] == "reeks"
        cur = [inf] * total_days
        ptr = [-1] * total_days
        best, arg = inf, -1
        for t in range(1, total_days):
            if series:
                best += REEKS_EXTRA_DAY
            if prev[t - 1] < best:
                best, arg = prev[t - 1], t - 1
            cur[t] = best + node[t]
            ptr[t] = arg
        back.append(ptr)
        prev = cur

    t = min(range(total_days), key=lambda i: prev[i])
    days = [t]
    for ptr in reversed(back):
        t = ptr[t]
        days.append(t)
    days.reverse()

    offsets = [0] * n
    for group, day in zip(groups, days):
        for i in group:
            offsets[i] = day
    return _to_datetimes(lo, offsets)


def moment_count(links: list[str]) -> int:
    return 1 + sum(1 for t in links[1:] if t != "moment") if links else 0


def period_label(start: Optional[date], end: Optional[date]) -> str:
    def fmt(d: Optional[date]) -> str:
        return f"{MONTH_NAMES[d.month][:3]} {d.year}" if d else "?"
    return f"{fmt(start)} – {fmt(end)}"


def describe_estimate(dt: datetime, album_name: str, start: date, end: date, note: str = "") -> str:
    # EXIF ImageDescription is ASCII-only, so no en dash here.
    period = period_label(start, end).replace("–", "-")
    note = f", {note}" if note else ""
    return f"Datum geschat: {MONTH_NAMES[dt.month]} {dt.year}{note} (album {album_name}, {period})"


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
    links: list[str] = field(default_factory=list)  # per photo: relation to the previous one
    seasons: list[str] = field(default_factory=list)  # per photo: season estimate
    holidays: list[str] = field(default_factory=list)  # per photo: recognised holiday or "geen"
    ages: list[Optional[int]] = field(default_factory=list)  # per photo: visible birthday age
    family: list[Person] = field(default_factory=list)
    order_errors: list[str] = field(default_factory=list)
    moved_starts: set[str] = field(default_factory=set)  # photos whose predecessor changed by a move

    @property
    def complete(self) -> bool:
        return self.start is not None and self.end is not None and self.start <= self.end

    @property
    def has_links(self) -> bool:
        return len(self.links) == len(self.photos) and all(self.links)

    @property
    def has_seasons(self) -> bool:
        return len(self.seasons) == len(self.photos) and any(s in SEASON_MONTH_COST for s in self.seasons)

    @property
    def has_holidays(self) -> bool:
        return len(self.holidays) == len(self.photos) and any(h in HOLIDAY_LABELS for h in self.holidays)

    def holiday_note(self, path: Path, when: date) -> str:
        """Name the holiday only when the placed date actually fits it (labels can be wrong)."""
        if not self.has_holidays:
            return ""
        i = self.photos.index(path)
        kind = self.holidays[i]
        if kind not in HOLIDAY_LABELS:
            return ""
        age = self.ages[i] if len(self.ages) == len(self.photos) else None
        margin = timedelta(days=7)
        windows = holiday_windows(kind, when - margin, when + margin, self.family, age)
        return HOLIDAY_LABELS[kind] if windows else ""

    def planned(self) -> list[tuple[Path, datetime]]:
        if not self.complete:
            return []
        if self.has_links and (self.has_seasons or self.has_holidays):
            ages = self.ages if len(self.ages) == len(self.photos) else None
            dts = place_with_seasons(
                self.links, self.seasons if self.has_seasons else ["onbekend"] * len(self.photos),
                self.start, self.end,
                holidays=self.holidays if self.has_holidays else None, ages=ages, family=self.family,
            )
        elif self.has_links:
            dts = spread_by_links(self.links, self.start, self.end)
        else:
            dts = spread_datetimes(len(self.photos), self.start, self.end)
        return list(zip(self.photos, dts))


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


MOVE_RE = re.compile(
    r"^\s*(\d+)\s*(?:-\s*(\d+))?\s+(?:(na|voor)\s+(\d+)|(vooraan|achteraan))\s*$", re.IGNORECASE,
)


def apply_order_moves(photos: list[Path], spec: str) -> tuple[list[Path], list[str]]:
    """Reorder by scan numbers, e.g. "168-201 na 004; 12 voor 3; 5-9 vooraan". Files keep their names."""
    order = list(photos)
    errors: list[str] = []
    for rule in (r.strip() for r in re.split(r"[;\n]", spec or "")):
        if not rule:
            continue
        m = MOVE_RE.match(rule)
        if not m:
            errors.append(f"niet begrepen: {rule}")
            continue
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        lo, hi = min(a, b), max(a, b)
        block = [p for p in order if (n := _seq_number(p)) is not None and lo <= n <= hi]
        if not block:
            errors.append(f"geen foto's met nummer {lo}–{hi}")
            continue
        rest = [p for p in order if p not in block]
        if m.group(5):
            idx = 0 if m.group(5).lower() == "vooraan" else len(rest)
        else:
            target = int(m.group(4))
            hits = [i for i, p in enumerate(rest) if _seq_number(p) == target]
            if not hits:
                errors.append(f"foto {target} niet gevonden (of zit in het blok zelf)")
                continue
            idx = hits[0] + 1 if m.group(3).lower() == "na" else hits[0]
        order = rest[:idx] + block + rest[idx:]
    return order, errors


def load_collection(
    folder: Path,
    rel_prefix: str = "",
    manual: Optional[dict[str, tuple[str, str]]] = None,
    moves: Optional[dict[str, str]] = None,
) -> list[CollectionAlbum]:
    """Albums directly under `folder`, with periods from manual input, folder name, then neighbours."""
    manual = manual or {}
    moves = moves or {}
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
        if moves.get(rel):
            original = album.photos
            album.photos, album.order_errors = apply_order_moves(original, moves[rel])
            before = {p.name: (original[i - 1].name if i else None) for i, p in enumerate(original)}
            album.moved_starts = {
                p.name for i, p in enumerate(album.photos)
                if (album.photos[i - 1].name if i else None) != before[p.name]
            }
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


def rewrite_exif(path: Path, mutate, backup_dir: Path, mtime: Optional[float]) -> None:
    """Change EXIF in place without re-encoding the image; keep the first original as backup."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / path.name
    if not backup.exists():
        shutil.copy2(path, backup)

    try:
        exif = piexif.load(str(path))
    except Exception:
        exif = {"0th": {}, "Exif": {}, "1st": {}, "GPS": {}}
    exif.pop("thumbnail", None)
    exif["1st"] = {}

    def dump(e: dict) -> bytes:
        mutate(e)
        return piexif.dump(e)

    try:
        exif_bytes = dump(exif)
    except Exception:
        # Phone EXIF sometimes holds tags piexif can't re-serialise; keep only what matters.
        exif_bytes = dump(_minimal_exif(exif))
    out = io.BytesIO()
    piexif.insert(exif_bytes, path.read_bytes(), out)
    # On SMB/NFS the server stamps mtime when buffered writes land; flush before setting it.
    with open(path, "r+b") as fh:
        fh.write(out.getvalue())
        fh.truncate()
        fh.flush()
        os.fsync(fh.fileno())

    if mtime is not None:
        try:
            os.utime(path, (mtime, mtime))
        except (OSError, OverflowError) as exc:
            logger.info("Kon bestandsdatum niet zetten voor %s: %s", path.name, exc)


def write_estimated_date(path: Path, dt: datetime, description: str, backup_dir: Path) -> None:
    """Write the estimated date + description into EXIF (no re-encode); keep the first original."""
    stamp = dt.strftime("%Y:%m:%d %H:%M:%S").encode()

    def mutate(e: dict) -> None:
        e["Exif"][piexif.ExifIFD.DateTimeOriginal] = stamp
        e["Exif"][piexif.ExifIFD.DateTimeDigitized] = stamp
        e["0th"][piexif.ImageIFD.DateTime] = stamp
        e["0th"][piexif.ImageIFD.ImageDescription] = description.encode("ascii", "replace")

    try:
        mtime: Optional[float] = time.mktime(dt.timetuple())
    except (OverflowError, ValueError):
        mtime = None
    rewrite_exif(path, mutate, backup_dir, mtime)


def apply_album(album: CollectionAlbum, backup_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Write spread dates for one album. Returns ({filename: iso datetime}, errors)."""
    written: dict[str, str] = {}
    errors: list[str] = []
    for path, dt in album.planned():
        try:
            description = describe_estimate(
                dt, album.name, album.start, album.end, album.holiday_note(path, dt.date()),
            )
            write_estimated_date(path, dt, description, backup_dir)
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

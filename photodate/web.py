import json
import logging
import os
import shutil
import subprocess
import threading
from datetime import date
from io import BytesIO
from pathlib import Path

import openai
from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from PIL import Image

from .analyzer import analyze_album_full
from .duplicates import find_duplicates
from .exif import read_exif_date, write_exif_date, parse_exif_date
from .faces import detect_faces_in_album
from .models import PhotoInfo
from .gphotos import (
    get_oauth_flow, load_credentials, save_credentials,
    get_service, list_all_media_items, match_local_to_google,
)
from .storage import AlbumData, FamilyMember, GlobalSettings, ICloudData, Milestone, STORAGE_DIR

logger = logging.getLogger(__name__)

app = FastAPI(title="Photodate")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
SKIP_DIRS = {"@eaDir", "#recycle", ".git", "_duplicates", "nietzelfgenomen"}
SYNOINDEX = Path("/usr/syno/bin/synoindex")


# --- Clean up stale "running" status files on startup ---
def _reset_stale_status():
    for status_file in STORAGE_DIR.glob("*_status.json"):
        try:
            data = json.loads(status_file.read_text())
            if data.get("running"):
                data["running"] = False
                data["error"] = "Afgebroken door herstart"
                status_file.write_text(json.dumps(data))
        except Exception:
            pass

try:
    _reset_stale_status()
except Exception:
    pass


# --- Helpers ---

def _get_photos_roots() -> list[Path]:
    raw = os.environ.get("PHOTOS_PATHS", os.environ.get("PHOTOS_PATH", "/photos"))
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def _get_all_albums() -> list[dict]:
    """Return list of album dicts with path, label, photo_count, never_organize, is_year_month."""
    albums = []
    for root in _get_photos_roots():
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            dirpath = Path(dirpath)
            photo_count = sum(1 for f in filenames if Path(f).suffix.lower() in EXTENSIONS)
            if photo_count > 0 and dirpath != root:
                rel = dirpath.relative_to(root)
                album_data = AlbumData.load(str(rel))
                albums.append({
                    "path": dirpath,
                    "label": str(rel),
                    "photo_count": photo_count,
                    "never_organize": album_data.never_organize,
                    "is_year_month": _is_year_month_folder(rel),
                })
    albums.sort(key=lambda x: x["label"])
    return albums


def _is_year_month_folder(rel: Path) -> bool:
    """Check if a relative path is a YYYY/MM folder directly in root."""
    parts = rel.parts
    return (
        len(parts) == 2
        and parts[0].isdigit() and len(parts[0]) == 4
        and parts[1].isdigit() and len(parts[1]) == 2
    )



def _find_album(album_rel: str) -> Path | None:
    for root in _get_photos_roots():
        candidate = root / album_rel
        if candidate.is_dir():
            return candidate
    return None


def _find_photos_root(album_folder: Path) -> Path | None:
    """Find which photos root contains the given album folder."""
    for root in _get_photos_roots():
        try:
            album_folder.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def _reindex_synology(filepath: Path) -> None:
    """Ask Synology to re-index a modified photo so Photos picks up new EXIF dates."""
    if not SYNOINDEX.exists():
        return
    try:
        subprocess.run(
            [str(SYNOINDEX), "-a", str(filepath)],
            timeout=10, capture_output=True,
        )
    except Exception as e:
        logger.warning(f"synoindex failed for {filepath}: {e}")


def _load_photos(folder: Path) -> list[PhotoInfo]:
    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    )
    return [
        PhotoInfo(path=f, index=i, original_exif_date=read_exif_date(f))
        for i, f in enumerate(files)
    ]


# --- Thumbnail endpoint ---

@app.get("/thumb/{album_rel:path}/__file__/{filename}")
async def thumbnail(album_rel: str, filename: str, size: int = 150):
    folder = _find_album(album_rel)
    if not folder:
        return Response(status_code=404)
    filepath = folder / filename
    if not filepath.is_file():
        return Response(status_code=404)
    img = Image.open(filepath)
    img.thumbnail((size, size))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


# --- Page: Home (album list, only directories) ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    albums = _get_all_albums()
    roots = _get_photos_roots()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "albums": albums,
        "photos_roots": roots,
    })


# --- Page: Organize (dedicated workflow page) ---

@app.get("/organize", response_class=HTMLResponse)
async def organize_page(request: Request):
    """Dedicated organize page: list all albums (fast, no EXIF reading)."""
    albums = _get_all_albums()

    organizable = []
    already_organized = []
    never_move = []

    for album in albums:
        if album["is_year_month"]:
            already_organized.append(album)
        elif album["never_organize"]:
            never_move.append(album)
        else:
            organizable.append(album)

    return templates.TemplateResponse("organize.html", {
        "request": request,
        "organizable": organizable,
        "already_organized": already_organized,
        "never_move": never_move,
    })


@app.get("/api/album/{album_rel:path}/exif-counts")
async def album_exif_counts(album_rel: str):
    """Return EXIF counts for a single album (used for lazy loading)."""
    folder = _find_album(album_rel)
    if not folder:
        return JSONResponse({"error": "not found"}, status_code=404)
    photos = _load_photos(folder)
    exif_count = sum(1 for p in photos if p.original_exif_date)
    return JSONResponse({
        "exif_count": exif_count,
        "no_exif_count": len(photos) - exif_count,
    })


@app.post("/organize-execute", response_class=HTMLResponse)
async def organize_execute_bulk(request: Request):
    """Move photos WITH EXIF from selected albums into YYYY/MM folders."""
    form = await request.form()
    album_labels = form.getlist("albums")

    total_moved = 0
    total_skipped = 0
    total_no_exif = 0
    all_errors = []
    albums_processed = []

    for album_rel in album_labels:
        folder = _find_album(album_rel)
        if not folder:
            all_errors.append(f"{album_rel}: map niet gevonden")
            continue

        photos_root = _find_photos_root(folder)
        if not photos_root:
            all_errors.append(f"{album_rel}: kan foto-root niet vinden")
            continue

        photos = _load_photos(folder)
        moved = 0
        skipped = 0
        no_exif = 0

        for photo in photos:
            if not photo.original_exif_date:
                no_exif += 1
                continue  # Leave photos without EXIF in place

            d = parse_exif_date(photo.original_exif_date)
            if not d:
                skipped += 1
                continue

            dest_dir = photos_root / str(d.year) / f"{d.month:02d}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / photo.path.name

            if photo.path.parent == dest_dir:
                skipped += 1
                continue

            if dest_path.exists():
                stem = photo.path.stem
                suffix = photo.path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            try:
                shutil.move(str(photo.path), str(dest_path))
                _reindex_synology(dest_path)
                moved += 1
            except Exception as e:
                all_errors.append(f"{album_rel}/{photo.path.name}: {e}")

        # Clean up empty source folder (ignore @eaDir etc.)
        if folder.exists():
            real_remaining = [f for f in folder.iterdir() if f.name not in SKIP_DIRS]
            source_empty = len(real_remaining) == 0
            if source_empty:
                try:
                    shutil.rmtree(str(folder))
                except Exception:
                    pass
        else:
            source_empty = True

        total_moved += moved
        total_skipped += skipped
        total_no_exif += no_exif
        albums_processed.append({
            "label": album_rel,
            "moved": moved,
            "skipped": skipped,
            "no_exif": no_exif,
            "removed": source_empty,
        })

    return templates.TemplateResponse("organize_done.html", {
        "request": request,
        "folder_name": f"{len(albums_processed)} albums",
        "moved_count": total_moved,
        "skipped_count": total_skipped,
        "no_exif_count": total_no_exif,
        "errors": all_errors,
        "albums_processed": albums_processed,
    })


@app.get("/organize-no-exif", response_class=HTMLResponse)
async def organize_no_exif_page(request: Request):
    """Show photos without EXIF per album with thumbnails and checkboxes."""
    album_labels = request.query_params.getlist("albums")
    albums_with_photos = []

    for album_rel in album_labels:
        folder = _find_album(album_rel)
        if not folder:
            continue
        photos = _load_photos(folder)
        no_exif_photos = [p for p in photos if not p.original_exif_date]
        if no_exif_photos:
            albums_with_photos.append({
                "label": album_rel,
                "photos": [{"filename": p.path.name} for p in no_exif_photos],
            })

    total_photos = sum(len(a["photos"]) for a in albums_with_photos)
    return templates.TemplateResponse("organize_no_exif.html", {
        "request": request,
        "albums": albums_with_photos,
        "total_photos": total_photos,
    })


@app.post("/organize-niet-zelf-genomen", response_class=HTMLResponse)
async def organize_niet_zelf_genomen(request: Request):
    """Move selected photos to /nietzelfgenomen/ folder."""
    form = await request.form()
    photo_keys = form.getlist("photos")  # values like "album_rel/filename"

    total_moved = 0
    all_errors = []
    cleaned_folders: set[str] = set()

    for photo_key in photo_keys:
        # Split into album_rel and filename
        sep = photo_key.rfind("/")
        if sep < 0:
            continue
        album_rel = photo_key[:sep]
        filename = photo_key[sep + 1:]

        folder = _find_album(album_rel)
        if not folder:
            all_errors.append(f"{photo_key}: map niet gevonden")
            continue

        source = folder / filename
        if not source.is_file():
            all_errors.append(f"{photo_key}: bestand niet gevonden")
            continue

        photos_root = _find_photos_root(folder)
        if not photos_root:
            all_errors.append(f"{photo_key}: kan foto-root niet vinden")
            continue

        nzg_dir = photos_root / "nietzelfgenomen"
        nzg_dir.mkdir(parents=True, exist_ok=True)

        dest_path = nzg_dir / filename
        if dest_path.exists():
            stem = source.stem
            suffix = source.suffix
            counter = 1
            while dest_path.exists():
                dest_path = nzg_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            shutil.move(str(source), str(dest_path))
            total_moved += 1
            cleaned_folders.add(album_rel)
        except Exception as e:
            all_errors.append(f"{photo_key}: {e}")

    # Clean up empty source folders
    for album_rel in cleaned_folders:
        folder = _find_album(album_rel)
        if folder and folder.exists():
            real_remaining = [f for f in folder.iterdir() if f.name not in SKIP_DIRS]
            if not real_remaining:
                try:
                    shutil.rmtree(str(folder))
                except Exception:
                    pass

    return templates.TemplateResponse("organize_done.html", {
        "request": request,
        "folder_name": "niet zelf genomen",
        "moved_count": total_moved,
        "skipped_count": 0,
        "no_exif_count": 0,
        "errors": all_errors,
        "albums_processed": [],
    })


# --- Toggle: never_organize flag ---

@app.post("/album/{album_rel:path}/toggle-never-organize")
async def toggle_never_organize(album_rel: str):
    album_data = AlbumData.load(album_rel)
    album_data.never_organize = not album_data.never_organize
    album_data.save()
    return JSONResponse({"never_organize": album_data.never_organize})


# --- iCloud photo matching ---

import time as _time

_photos_index_cache: set[str] | None = None
_photos_index_timestamp: float = 0
_photos_index_lock = threading.Lock()
_PHOTOS_INDEX_TTL = 300  # Cache for 5 minutes


def _build_photos_filename_index() -> set[str]:
    """Build a set of all photo filenames across all PHOTOS_PATHS (cached 5 min, thread-safe)."""
    global _photos_index_cache, _photos_index_timestamp
    now = _time.time()
    if _photos_index_cache is not None and (now - _photos_index_timestamp) < _PHOTOS_INDEX_TTL:
        return _photos_index_cache

    with _photos_index_lock:
        # Double-check after acquiring lock
        now = _time.time()
        if _photos_index_cache is not None and (now - _photos_index_timestamp) < _PHOTOS_INDEX_TTL:
            return _photos_index_cache

        index: set[str] = set()
        for root in _get_photos_roots():
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for f in filenames:
                    if Path(f).suffix.lower() in EXTENSIONS:
                        index.add(f.lower())
        _photos_index_cache = index
        _photos_index_timestamp = now
        logger.info(f"Photos filename index built: {len(index)} files")
        return index


def _get_icloud_folders(icloud_path: Path) -> list[dict]:
    """Scan iCloud directory for YYYY/MM folder structure. Returns sorted list."""
    folders = []
    if not icloud_path.exists():
        return folders
    for year_dir in sorted(icloud_path.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit() or len(year_dir.name) != 4:
            continue
        year = year_dir.name
        # Check for month subdirs
        has_months = False
        for month_dir in sorted(year_dir.iterdir()):
            if month_dir.is_dir() and month_dir.name.isdigit() and len(month_dir.name) == 2:
                has_months = True
                photo_count = sum(
                    1 for f in month_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in EXTENSIONS
                )
                if photo_count > 0:
                    folders.append({
                        "year": year,
                        "month": month_dir.name,
                        "path": month_dir,
                        "photo_count": photo_count,
                    })
        # If no month subdirs, treat year folder itself as a single entry
        if not has_months:
            photo_count = sum(
                1 for f in year_dir.iterdir()
                if f.is_file() and f.suffix.lower() in EXTENSIONS
            )
            if photo_count > 0:
                folders.append({
                    "year": year,
                    "month": "00",
                    "path": year_dir,
                    "photo_count": photo_count,
                })
    return folders


@app.get("/icloud", response_class=HTMLResponse)
async def icloud_dashboard(request: Request):
    """iCloud matching dashboard: list all year/month folders."""
    settings = GlobalSettings.load()
    icloud_path = Path(settings.icloud_photos_path) if settings.icloud_photos_path else None

    if not icloud_path or not icloud_path.exists():
        return templates.TemplateResponse("icloud.html", {
            "request": request,
            "folders": [],
            "no_path": True,
        })

    folders = _get_icloud_folders(icloud_path)

    # Pre-build filename index in background so first API call is fast
    threading.Thread(target=_build_photos_filename_index, daemon=True).start()

    return templates.TemplateResponse("icloud.html", {
        "request": request,
        "folders": folders,
        "no_path": False,
    })


@app.get("/api/icloud/{year}/{month}/counts")
async def icloud_counts(year: str, month: str):
    """Return match counts for one iCloud year/month folder (lazy loading)."""
    settings = GlobalSettings.load()
    if not settings.icloud_photos_path:
        return JSONResponse({"error": "no icloud path"}, status_code=400)

    icloud_path = Path(settings.icloud_photos_path)
    if month == "00":
        folder = icloud_path / year
    else:
        folder = icloud_path / year / month

    if not folder.exists():
        return JSONResponse({"error": "folder not found"}, status_code=404)

    # Get all iCloud photos in this folder
    icloud_files = [
        f.name for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    ]
    total = len(icloud_files)
    if total == 0:
        return JSONResponse({"total": 0, "matched": 0, "dismissed": 0, "unmatched": 0, "pct": 100})

    # Build filename index of user's photos
    index = _build_photos_filename_index()

    # Load dismissed data
    icloud_data = ICloudData.load()
    key = f"{year}/{month}"
    dismissed_set = set(icloud_data.dismissed.get(key, []))

    matched = 0
    dismissed = 0
    unmatched = 0

    for fname in icloud_files:
        if fname.lower() in index:
            matched += 1
        elif fname in dismissed_set:
            dismissed += 1
        else:
            unmatched += 1

    pct = round((matched + dismissed) / total * 100) if total > 0 else 100
    return JSONResponse({
        "total": total,
        "matched": matched,
        "dismissed": dismissed,
        "unmatched": unmatched,
        "pct": pct,
    })


@app.get("/icloud-thumb/{year}/{month}/{filename}")
async def icloud_thumbnail(year: str, month: str, filename: str, size: int = 150):
    """Serve thumbnail for an iCloud photo."""
    settings = GlobalSettings.load()
    if not settings.icloud_photos_path:
        return Response(status_code=404)

    icloud_path = Path(settings.icloud_photos_path)
    if month == "00":
        filepath = icloud_path / year / filename
    else:
        filepath = icloud_path / year / month / filename

    if not filepath.is_file():
        return Response(status_code=404)

    img = Image.open(filepath)
    img.thumbnail((size, size))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.get("/icloud/{year}/{month}/review", response_class=HTMLResponse)
async def icloud_review(request: Request, year: str, month: str):
    """Show unmatched iCloud photos with thumbnails for keep/dismiss."""
    settings = GlobalSettings.load()
    if not settings.icloud_photos_path:
        return HTMLResponse("iCloud pad niet ingesteld", status_code=400)

    icloud_path = Path(settings.icloud_photos_path)
    if month == "00":
        folder = icloud_path / year
    else:
        folder = icloud_path / year / month

    if not folder.exists():
        return HTMLResponse("Map niet gevonden", status_code=404)

    icloud_files = sorted(
        f.name for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    )

    index = _build_photos_filename_index()
    icloud_data = ICloudData.load()
    key = f"{year}/{month}"
    dismissed_set = set(icloud_data.dismissed.get(key, []))

    # Only show unmatched (not in user photos AND not dismissed)
    unmatched = [f for f in icloud_files if f.lower() not in index and f not in dismissed_set]

    return templates.TemplateResponse("icloud_review.html", {
        "request": request,
        "year": year,
        "month": month,
        "photos": unmatched,
        "total_unmatched": len(unmatched),
    })


@app.post("/icloud/{year}/{month}/action", response_class=HTMLResponse)
async def icloud_action(request: Request, year: str, month: str):
    """Process keep/dismiss actions for iCloud photos."""
    form = await request.form()
    action = form.get("action", "")
    photo_names = form.getlist("photos")

    if not photo_names:
        return RedirectResponse(url=f"/icloud/{year}/{month}/review", status_code=303)

    settings = GlobalSettings.load()
    if not settings.icloud_photos_path:
        return HTMLResponse("iCloud pad niet ingesteld", status_code=400)

    icloud_path = Path(settings.icloud_photos_path)
    if month == "00":
        folder = icloud_path / year
    else:
        folder = icloud_path / year / month

    if action == "dismiss":
        # Mark as not important
        icloud_data = ICloudData.load()
        key = f"{year}/{month}"
        existing = set(icloud_data.dismissed.get(key, []))
        existing.update(photo_names)
        icloud_data.dismissed[key] = sorted(existing)
        icloud_data.save()

    elif action == "keep":
        # Copy to YYYY/MM in user's photos
        photos_roots = _get_photos_roots()
        if not photos_roots:
            return HTMLResponse("Geen foto-mappen ingesteld", status_code=400)
        photos_root = photos_roots[0]
        dest_dir = photos_root / year / (month if month != "00" else "01")
        dest_dir.mkdir(parents=True, exist_ok=True)

        errors = []
        for fname in photo_names:
            source = folder / fname
            if not source.is_file():
                errors.append(f"{fname}: niet gevonden")
                continue
            dest_path = dest_dir / fname
            if dest_path.exists():
                stem = source.stem
                suffix = source.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            try:
                shutil.copy2(str(source), str(dest_path))
                _reindex_synology(dest_path)
            except Exception as e:
                errors.append(f"{fname}: {e}")

    return RedirectResponse(url=f"/icloud/{year}/{month}/review", status_code=303)


# --- Page: Global Settings (family members) ---

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    settings = GlobalSettings.load()
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "settings": settings,
        "google_connected": load_credentials() is not None,
    })


@app.post("/settings", response_class=HTMLResponse)
async def settings_save(request: Request):
    form = await request.form()
    members = []
    i = 0
    while f"name_{i}" in form:
        name = form.get(f"name_{i}", "").strip()
        birthdate = form.get(f"birthdate_{i}", "").strip()
        notes = form.get(f"notes_{i}", "").strip()
        if name:
            members.append(FamilyMember(name=name, birthdate=birthdate, notes=notes))
        i += 1

    # Handle Google credentials file upload
    old_settings = GlobalSettings.load()
    google_creds_path = old_settings.google_credentials_path

    creds_file = form.get("google_credentials")
    if creds_file and hasattr(creds_file, "read"):
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        dest = STORAGE_DIR / "google_credentials.json"
        content = await creds_file.read()
        if content:
            dest.write_bytes(content)
            google_creds_path = str(dest)

    icloud_photos_path = form.get("icloud_photos_path", "").strip()

    settings = GlobalSettings(
        family_members=members,
        google_credentials_path=google_creds_path,
        icloud_photos_path=icloud_photos_path,
    )
    settings.save()
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "settings": settings,
        "google_connected": load_credentials() is not None,
        "saved": True,
    })


# --- Page: Date review (step 3) --- (must be before catch-all album route)

@app.get("/album/{album_rel:path}/dates", response_class=HTMLResponse)
async def date_review_page(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)
    photos = _load_photos(folder)
    album_data = AlbumData.load(album_rel)
    return templates.TemplateResponse("dates.html", {
        "request": request,
        "album_rel": album_rel,
        "photos": photos,
        "album_data": album_data,
    })


# --- Page: Analyze (step 2: face detection + AI analysis) ---

@app.post("/album/{album_rel:path}/analyze", response_class=HTMLResponse)
async def album_analyze(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    form = await request.form()
    limit = int(form.get("limit", 0)) or None
    only_missing = form.get("only_missing") == "1"

    photos = _load_photos(folder)
    if only_missing:
        photos = [p for p in photos if not p.original_exif_date]
    if limit:
        photos = photos[:limit]

    album_data = AlbumData.load(album_rel)
    settings = GlobalSettings.load()

    # Face detection (skip — disabled on Synology)
    face_results = detect_faces_in_album([p.path for p in photos])

    # AI date analysis
    try:
        client = openai.OpenAI()
        album_data = analyze_album_full(photos, album_data, settings, client)
    except Exception as e:
        return HTMLResponse(
            f"<h2>Fout bij AI analyse</h2><pre>{type(e).__name__}: {e}</pre>"
            f"<p><a href='/album/{album_rel}'>Terug</a></p>",
            status_code=500,
        )

    return templates.TemplateResponse("results.html", {
        "request": request,
        "album_rel": album_rel,
        "photos": photos,
        "album_data": album_data,
        "face_clusters": face_results["clusters"],
        "faces_per_photo": face_results.get("per_photo", {}),
    })


# --- Page: Save face labels (from results page) ---

@app.post("/album/{album_rel:path}/faces", response_class=HTMLResponse)
async def save_face_labels(request: Request, album_rel: str):
    form = await request.form()
    album_data = AlbumData.load(album_rel)

    for key, value in form.items():
        if key.startswith("face_label_"):
            face_id = key.replace("face_label_", "")
            if value.strip():
                album_data.face_labels[face_id] = value.strip()

    album_data.save()

    # Redirect to date review page
    folder = _find_album(album_rel)
    photos = _load_photos(folder)
    return templates.TemplateResponse("dates.html", {
        "request": request,
        "album_rel": album_rel,
        "photos": photos,
        "album_data": album_data,
    })


@app.post("/album/{album_rel:path}/save", response_class=HTMLResponse)
async def album_context_save(request: Request, album_rel: str):
    form = await request.form()
    album_data = AlbumData.load(album_rel)
    album_data.start_date = form.get("start_date", "").strip()
    album_data.end_date = form.get("end_date", "").strip()
    album_data.context_notes = form.get("context_notes", "").strip()

    # Parse milestones
    milestones = []
    i = 0
    while f"ms_photo_{i}" in form:
        photo_idx = form.get(f"ms_photo_{i}", "").strip()
        ms_date = form.get(f"ms_date_{i}", "").strip()
        ms_desc = form.get(f"ms_desc_{i}", "").strip()
        if photo_idx and ms_date and ms_desc:
            milestones.append(Milestone(
                photo_index=int(photo_idx),
                date=ms_date,
                description=ms_desc,
            ))
        i += 1
    album_data.milestones = milestones
    album_data.save()

    folder = _find_album(album_rel)
    photos = _load_photos(folder)
    return templates.TemplateResponse("album_context.html", {
        "request": request,
        "album_rel": album_rel,
        "album_data": album_data,
        "photo_count": len(photos),
        "missing_exif": sum(1 for p in photos if not p.original_exif_date),
        "folder_name": folder.name,
        "saved": True,
    })


# --- Page: Apply dates (step 4) ---

@app.post("/album/{album_rel:path}/apply", response_class=HTMLResponse)
async def apply_dates(request: Request, album_rel: str):
    form = await request.form()
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    count = 0
    reindexed = 0
    for key, value in form.items():
        if key.startswith("date_") and value:
            filename = key[5:]
            filepath = folder / filename
            if filepath.is_file():
                try:
                    dt = date.fromisoformat(value)
                    write_exif_date(filepath, dt, backup=True)
                    count += 1
                    _reindex_synology(filepath)
                    reindexed += 1
                except (ValueError, Exception):
                    pass

    return templates.TemplateResponse("applied.html", {
        "request": request,
        "album_rel": album_rel,
        "folder_name": folder.name,
        "count": count,
        "reindexed": reindexed if SYNOINDEX.exists() else None,
    })


# --- Page: Duplicate detection ---

@app.get("/album/{album_rel:path}/duplicates", response_class=HTMLResponse)
async def album_duplicates(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    photos = _load_photos(folder)
    album_data = AlbumData.load(album_rel)

    # Find duplicates using perceptual hashing
    photo_paths = [p.path for p in photos]
    groups, updated_hashes = find_duplicates(
        photo_paths, cached_hashes=album_data.photo_hashes
    )

    # Save updated hashes to cache
    album_data.photo_hashes = updated_hashes
    album_data.save()

    # Check if _duplicates folder exists and has files
    dup_folder = folder / "_duplicates"
    dup_files = []
    if dup_folder.is_dir():
        dup_files = sorted(
            f.name for f in dup_folder.iterdir()
            if f.is_file() and f.suffix.lower() in EXTENSIONS
        )

    return templates.TemplateResponse("duplicates.html", {
        "request": request,
        "album_rel": album_rel,
        "folder_name": folder.name,
        "groups": groups,
        "dup_files": dup_files,
        "photo_count": len(photos),
    })


@app.post("/album/{album_rel:path}/duplicates/move", response_class=HTMLResponse)
async def move_duplicates(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    form = await request.form()
    dup_folder = folder / "_duplicates"

    moved = 0
    # For each group, the "keep" radio has the filename to keep.
    # All other files in that group should be moved.
    for key, value in form.items():
        if key.startswith("move_"):
            filename = key[5:]
            filepath = folder / filename
            if filepath.is_file():
                dup_folder.mkdir(exist_ok=True)
                shutil.move(str(filepath), str(dup_folder / filename))
                moved += 1

    # Redirect back to duplicates page
    from fastapi.responses import RedirectResponse
    return RedirectResponse(
        url=f"/album/{album_rel}/duplicates?moved={moved}",
        status_code=303,
    )


@app.post("/album/{album_rel:path}/duplicates/cleanup", response_class=HTMLResponse)
async def cleanup_duplicates(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    dup_folder = folder / "_duplicates"
    removed = 0
    if dup_folder.is_dir():
        for f in dup_folder.iterdir():
            if f.is_file():
                f.unlink()
                removed += 1
        dup_folder.rmdir()

    from fastapi.responses import RedirectResponse
    return RedirectResponse(
        url=f"/album/{album_rel}/duplicates?cleaned={removed}",
        status_code=303,
    )


# --- Page: Global duplicate detection ---

DUPSCAN_STATUS_PATH = STORAGE_DIR / "dupscan_status.json"
DUPSCAN_RESULT_PATH = STORAGE_DIR / "dupscan_result.json"
DUPSCAN_CACHE_PATH = STORAGE_DIR / "dupscan_hash_cache.json"
_dupscan_lock = threading.Lock()


def _run_global_dupscan():
    """Background task: scan all albums for duplicates."""
    try:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        DUPSCAN_STATUS_PATH.write_text(json.dumps({
            "running": True, "phase": "collecting", "detail": "Foto's verzamelen...",
            "progress": 0, "total": 0,
        }))

        # Collect all photo paths across all albums
        all_paths = []
        path_to_album = {}
        for root in _get_photos_roots():
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                dirpath_p = Path(dirpath)
                for fname in filenames:
                    filepath = dirpath_p / fname
                    if filepath.suffix.lower() in EXTENSIONS:
                        all_paths.append(filepath)
                        try:
                            rel = dirpath_p.relative_to(root)
                        except ValueError:
                            rel = dirpath_p
                        path_to_album[str(filepath)] = str(rel)

        total = len(all_paths)

        # Load hash cache from previous scans
        hash_cache = {}
        if DUPSCAN_CACHE_PATH.exists():
            try:
                hash_cache = json.loads(DUPSCAN_CACHE_PATH.read_text())
            except Exception:
                pass

        cached = sum(1 for p in all_paths if str(p) in hash_cache)
        DUPSCAN_STATUS_PATH.write_text(json.dumps({
            "running": True, "phase": "hashing",
            "detail": f"0/{total} foto's gehasht ({cached} uit cache)",
            "progress": 0, "total": total,
        }))

        groups, updated_cache = find_duplicates(
            all_paths, cached_hashes=hash_cache, threshold=8,
            progress_callback=lambda i, t: DUPSCAN_STATUS_PATH.write_text(json.dumps({
                "running": True, "phase": "hashing",
                "detail": f"{i}/{t} foto's gehasht",
                "progress": i, "total": t,
            })))

        # Save hash cache for next scan
        try:
            DUPSCAN_CACHE_PATH.write_text(json.dumps(updated_cache))
        except Exception:
            pass

        # Serialize groups with album info
        result_groups = []
        for g in groups:
            photos = []
            for p in g.photos:
                photos.append({
                    "filename": p.filename,
                    "path": str(p.path),
                    "album": path_to_album.get(str(p.path), ""),
                    "size_bytes": p.size_bytes,
                    "width": p.width,
                    "height": p.height,
                    "exif_date": p.exif_date,
                })
            result_groups.append({"id": g.id, "photos": photos})

        DUPSCAN_RESULT_PATH.write_text(json.dumps({
            "groups": result_groups,
            "total_photos": total,
            "total_groups": len(result_groups),
        }))
        DUPSCAN_STATUS_PATH.write_text(json.dumps({"running": False, "done": True}))

    except Exception as e:
        logger.exception("Global duplicate scan failed")
        DUPSCAN_STATUS_PATH.write_text(json.dumps({"running": False, "error": str(e)}))


@app.get("/duplicates", response_class=HTMLResponse)
async def global_duplicates(request: Request, moved: int = 0):
    result = None
    if DUPSCAN_RESULT_PATH.exists():
        try:
            result = json.loads(DUPSCAN_RESULT_PATH.read_text())
        except Exception:
            pass

    scan_running = False
    if DUPSCAN_STATUS_PATH.exists():
        try:
            status = json.loads(DUPSCAN_STATUS_PATH.read_text())
            scan_running = status.get("running", False)
        except Exception:
            pass

    return templates.TemplateResponse("global_duplicates.html", {
        "request": request,
        "result": result,
        "scan_running": scan_running,
        "moved": moved,
    })


@app.post("/duplicates/scan")
async def global_duplicates_scan():
    with _dupscan_lock:
        if DUPSCAN_STATUS_PATH.exists():
            try:
                status = json.loads(DUPSCAN_STATUS_PATH.read_text())
                if status.get("running"):
                    return JSONResponse({"ok": False, "message": "Scan loopt al"})
            except Exception:
                pass
        if DUPSCAN_RESULT_PATH.exists():
            DUPSCAN_RESULT_PATH.unlink()
        DUPSCAN_STATUS_PATH.write_text(json.dumps({"running": True, "phase": "starting", "detail": "Start..."}))

    thread = threading.Thread(target=_run_global_dupscan, daemon=True)
    thread.start()
    return JSONResponse({"ok": True})


@app.get("/duplicates/status")
async def global_duplicates_status():
    if not DUPSCAN_STATUS_PATH.exists():
        return JSONResponse({"running": False})
    try:
        return JSONResponse(json.loads(DUPSCAN_STATUS_PATH.read_text()))
    except Exception:
        return JSONResponse({"running": False})


@app.post("/duplicates/cancel")
async def global_duplicates_cancel():
    if DUPSCAN_STATUS_PATH.exists():
        DUPSCAN_STATUS_PATH.write_text(json.dumps({"running": False, "error": "Geannuleerd"}))
    return JSONResponse({"ok": True})


@app.post("/duplicates/move")
async def global_duplicates_move(request: Request):
    form = await request.form()
    moved = 0

    # Use the first photos root as central duplicates location
    roots = _get_photos_roots()
    central_dup_folder = roots[0] / "_duplicates" if roots else Path("/data/_duplicates")
    central_dup_folder.mkdir(parents=True, exist_ok=True)

    for key, value in form.items():
        if key.startswith("move_"):
            filepath = Path(value)  # Full path stored in form value
            if filepath.is_file():
                dest = central_dup_folder / filepath.name
                # Handle name collisions
                if dest.exists():
                    stem = filepath.stem
                    suffix = filepath.suffix
                    i = 1
                    while dest.exists():
                        dest = central_dup_folder / f"{stem}_{i}{suffix}"
                        i += 1
                shutil.move(str(filepath), str(dest))
                moved += 1

    # Clear cached results after moving — a new scan will use the hash cache
    if DUPSCAN_RESULT_PATH.exists():
        DUPSCAN_RESULT_PATH.unlink()

    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/duplicates?moved={moved}", status_code=303)


# --- Page: Google Photos verification ---

VERIFY_STATUS_PATH = STORAGE_DIR / "verify_status.json"
VERIFY_RESULT_PATH = STORAGE_DIR / "verify_result.json"
_verify_lock = threading.Lock()


def _update_verify_status(phase: str, detail: str = "", progress: int = 0, total: int = 0):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_STATUS_PATH.write_text(json.dumps({
        "running": True,
        "phase": phase,
        "detail": detail,
        "progress": progress,
        "total": total,
    }))


def _run_verify_scan():
    """Background task: scan local photos and match against Google Photos."""
    try:
        _update_verify_status("local", "Lokale foto's scannen...")
        creds = load_credentials()
        if not creds:
            VERIFY_STATUS_PATH.write_text(json.dumps({"running": False, "error": "Niet geautoriseerd"}))
            return

        service = get_service(creds)
        local_photos = []
        # First count files for progress
        all_files = []
        for root in _get_photos_roots():
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for fname in filenames:
                    filepath = Path(dirpath) / fname
                    if filepath.suffix.lower() in EXTENSIONS:
                        all_files.append(filepath)

        total_files = len(all_files)
        for i, filepath in enumerate(all_files):
            if i % 100 == 0:
                _update_verify_status("local", f"{i}/{total_files} foto's gescand", i, total_files)
            fname = filepath.name
            exif_date = read_exif_date(filepath)
            try:
                img = Image.open(filepath)
                w, h = img.size
            except Exception:
                w, h = 0, 0

            year = "onbekend"
            if exif_date and len(exif_date) >= 4:
                year = exif_date[:4]

            local_photos.append({
                "filename": fname,
                "path": str(filepath),
                "exif_date": exif_date,
                "width": w,
                "height": h,
                "year": year,
            })

        _update_verify_status("google", "Google Photos ophalen...", total_files, total_files)

        try:
            google_items = list_all_media_items(service)
        except Exception as e:
            VERIFY_STATUS_PATH.write_text(json.dumps({
                "running": False, "error": f"Google Photos fout: {e}",
            }))
            return

        _update_verify_status("matching", "Foto's matchen...", 0, 0)
        result = match_local_to_google(local_photos, google_items)
        result["google_count"] = len(google_items)

        # Serialize result (remove non-JSON-serializable fields)
        VERIFY_RESULT_PATH.write_text(json.dumps(result))
        VERIFY_STATUS_PATH.write_text(json.dumps({"running": False, "done": True}))

    except Exception as e:
        logger.exception("Verify scan failed")
        VERIFY_STATUS_PATH.write_text(json.dumps({"running": False, "error": str(e)}))


@app.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request):
    settings = GlobalSettings.load()
    creds = load_credentials()

    if not creds:
        return templates.TemplateResponse("verify_report.html", {
            "request": request,
            "needs_auth": True,
            "has_credentials": bool(settings.google_credentials_path),
        })

    # Check if we have cached results
    result = None
    if VERIFY_RESULT_PATH.exists():
        try:
            result = json.loads(VERIFY_RESULT_PATH.read_text())
        except Exception:
            pass

    # Check if scan is running
    scan_running = False
    if VERIFY_STATUS_PATH.exists():
        try:
            status = json.loads(VERIFY_STATUS_PATH.read_text())
            scan_running = status.get("running", False)
        except Exception:
            pass

    return templates.TemplateResponse("verify_report.html", {
        "request": request,
        "result": result,
        "google_count": result.get("google_count", 0) if result else 0,
        "scan_running": scan_running,
    })


@app.post("/verify/scan")
async def verify_start_scan():
    with _verify_lock:
        # Check if already running
        if VERIFY_STATUS_PATH.exists():
            try:
                status = json.loads(VERIFY_STATUS_PATH.read_text())
                if status.get("running"):
                    return JSONResponse({"ok": False, "message": "Scan loopt al"})
            except Exception:
                pass

        # Clear old results
        if VERIFY_RESULT_PATH.exists():
            VERIFY_RESULT_PATH.unlink()
        _update_verify_status("starting", "Scan wordt gestart...")

    thread = threading.Thread(target=_run_verify_scan, daemon=True)
    thread.start()
    return JSONResponse({"ok": True})


@app.get("/verify/status")
async def verify_status():
    if not VERIFY_STATUS_PATH.exists():
        return JSONResponse({"running": False})
    try:
        return JSONResponse(json.loads(VERIFY_STATUS_PATH.read_text()))
    except Exception:
        return JSONResponse({"running": False})


@app.get("/verify/auth", response_class=HTMLResponse)
async def verify_auth(request: Request):
    settings = GlobalSettings.load()
    if not settings.google_credentials_path:
        return HTMLResponse(
            "<h2>Geen Google credentials</h2>"
            "<p>Upload eerst je Google OAuth2 credentials JSON via <a href='/settings'>Instellingen</a>.</p>",
        )

    base_url = str(request.base_url).rstrip("/")
    redirect_uri = f"{base_url}/verify/callback"
    flow = get_oauth_flow(settings.google_credentials_path, redirect_uri)
    auth_url, state = flow.authorization_url(prompt="consent", access_type="offline")

    # Save code_verifier for PKCE (needed in callback)
    verifier_path = STORAGE_DIR / "oauth_code_verifier.txt"
    verifier_path.write_text(flow.code_verifier or "")

    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=auth_url)


@app.get("/verify/callback", response_class=HTMLResponse)
async def verify_callback(request: Request):
    settings = GlobalSettings.load()
    base_url = str(request.base_url).rstrip("/")
    redirect_uri = f"{base_url}/verify/callback"
    flow = get_oauth_flow(settings.google_credentials_path, redirect_uri)

    # Restore PKCE code_verifier from auth step
    verifier_path = STORAGE_DIR / "oauth_code_verifier.txt"
    if verifier_path.exists():
        flow.code_verifier = verifier_path.read_text() or None
        verifier_path.unlink()

    # Ensure authorization_response uses same scheme as redirect_uri
    auth_response = str(request.url)
    if redirect_uri.startswith("https://") and auth_response.startswith("http://"):
        auth_response = "https://" + auth_response[len("http://"):]

    try:
        flow.fetch_token(authorization_response=auth_response)
        save_credentials(flow.credentials)
    except Exception as e:
        logger.exception("OAuth callback failed")
        return HTMLResponse(
            f"<h2>OAuth fout</h2><p>{e}</p>"
            "<p><a href='/verify/auth'>Probeer opnieuw</a></p>",
            status_code=500,
        )

    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/verify")


# --- Page: Album context (step 1) --- (catch-all, must be LAST)

@app.get("/album/{album_rel:path}", response_class=HTMLResponse)
async def album_context_page(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)
    album_data = AlbumData.load(album_rel)
    photos = _load_photos(folder)
    missing_exif = sum(1 for p in photos if not p.original_exif_date)
    is_year_month = _is_year_month_folder(Path(album_rel))
    return templates.TemplateResponse("album_context.html", {
        "request": request,
        "album_rel": album_rel,
        "album_data": album_data,
        "photo_count": len(photos),
        "missing_exif": missing_exif,
        "folder_name": folder.name,
        "is_year_month": is_year_month,
    })

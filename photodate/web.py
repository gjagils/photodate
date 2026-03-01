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
from fastapi.responses import HTMLResponse, JSONResponse, Response
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
from .storage import AlbumData, FamilyMember, GlobalSettings, Milestone, STORAGE_DIR

logger = logging.getLogger(__name__)

app = FastAPI(title="Photodate")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
SKIP_DIRS = {"@eaDir", "#recycle", ".git", "_duplicates"}
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
    """Return list of album dicts with path, label, photo_count, exif_complete."""
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
                    "exif_complete": album_data.exif_complete,
                })
    albums.sort(key=lambda x: x["label"])
    return albums


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


# --- Toggle: EXIF complete flag ---

@app.post("/album/{album_rel:path}/toggle-exif-complete")
async def toggle_exif_complete(album_rel: str):
    album_data = AlbumData.load(album_rel)
    album_data.exif_complete = not album_data.exif_complete
    album_data.save()
    return JSONResponse({"exif_complete": album_data.exif_complete})


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

    settings = GlobalSettings(
        family_members=members,
        google_credentials_path=google_creds_path,
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


# --- Page: Organize photos into YYYY/MM folders ---

@app.get("/album/{album_rel:path}/organize", response_class=HTMLResponse)
async def organize_preview(request: Request, album_rel: str):
    """Preview: show photos grouped by YYYY/MM based on EXIF dates."""
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    photos = _load_photos(folder)
    missing_exif = sum(1 for p in photos if not p.original_exif_date)

    if missing_exif > 0:
        return HTMLResponse(
            f"<h2>Niet alle foto's hebben een EXIF-datum</h2>"
            f"<p>{missing_exif} van {len(photos)} foto's mist een EXIF-datum. "
            f"Analyseer eerst het album zodat alle foto's een datum hebben.</p>"
            f"<p><a href='/album/{album_rel}'>Terug naar album</a></p>",
            status_code=400,
        )

    # Group photos by YYYY/MM
    groups: dict[str, list[dict]] = {}
    for photo in photos:
        d = parse_exif_date(photo.original_exif_date)
        if d:
            key = f"{d.year}/{d.month:02d}"
            if key not in groups:
                groups[key] = []
            groups[key].append({
                "filename": photo.path.name,
                "path": str(photo.path),
                "exif_date": photo.original_exif_date,
                "date": d.isoformat(),
            })

    # Sort groups by year/month
    sorted_groups = sorted(groups.items())

    # Find the photos root for destination path display
    photos_root = _find_photos_root(folder)
    dest_base = str(photos_root) if photos_root else "?"

    return templates.TemplateResponse("organize_preview.html", {
        "request": request,
        "album_rel": album_rel,
        "folder_name": folder.name,
        "photo_count": len(photos),
        "groups": sorted_groups,
        "group_count": len(sorted_groups),
        "dest_base": dest_base,
    })


@app.post("/album/{album_rel:path}/organize", response_class=HTMLResponse)
async def organize_execute(request: Request, album_rel: str):
    """Move photos into YYYY/MM folders in the photos root."""
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    photos_root = _find_photos_root(folder)
    if not photos_root:
        return HTMLResponse("Kan de foto-root niet vinden", status_code=500)

    photos = _load_photos(folder)
    missing_exif = sum(1 for p in photos if not p.original_exif_date)
    if missing_exif > 0:
        return HTMLResponse("Niet alle foto's hebben een EXIF-datum", status_code=400)

    moved_count = 0
    skipped_count = 0
    errors = []

    for photo in photos:
        d = parse_exif_date(photo.original_exif_date)
        if not d:
            skipped_count += 1
            continue

        dest_dir = photos_root / str(d.year) / f"{d.month:02d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / photo.path.name

        # Skip if already in the right place
        if photo.path.parent == dest_dir:
            skipped_count += 1
            continue

        # Handle filename collision
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
            moved_count += 1
        except Exception as e:
            errors.append(f"{photo.path.name}: {e}")
            logger.warning(f"Failed to move {photo.path}: {e}")

    # Clean up empty source folder
    source_empty = not any(folder.iterdir())
    if source_empty:
        try:
            folder.rmdir()
            logger.info(f"Removed empty source folder: {folder}")
        except Exception as e:
            logger.warning(f"Could not remove {folder}: {e}")

    return templates.TemplateResponse("organize_done.html", {
        "request": request,
        "album_rel": album_rel,
        "folder_name": folder.name,
        "moved_count": moved_count,
        "skipped_count": skipped_count,
        "errors": errors,
        "source_removed": source_empty,
    })


# --- Page: Album context (step 1) --- (catch-all, must be LAST)

@app.get("/album/{album_rel:path}", response_class=HTMLResponse)
async def album_context_page(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)
    album_data = AlbumData.load(album_rel)
    photos = _load_photos(folder)
    missing_exif = sum(1 for p in photos if not p.original_exif_date)
    return templates.TemplateResponse("album_context.html", {
        "request": request,
        "album_rel": album_rel,
        "album_data": album_data,
        "photo_count": len(photos),
        "missing_exif": missing_exif,
        "folder_name": folder.name,
    })

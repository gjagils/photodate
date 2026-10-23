import asyncio
import json
import logging
import os
import shutil
import traceback
from datetime import date
from io import BytesIO
from pathlib import Path

import openai
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from PIL import Image

from . import collection as coll
from .analyzer import analyze_album_full
from .exif import read_exif_date, write_exif_date
from .faces import detect_faces_in_album
from .models import PhotoInfo
from .storage import (
    STORAGE_DIR,
    AlbumData,
    FaceProfile,
    FamilyMember,
    GlobalSettings,
    Milestone,
)

logger = logging.getLogger(__name__)

# HEIC support — optional dependency
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".heic", ".heif"}
    logger.info("HEIC/HEIF ondersteuning ingeschakeld")
except ImportError:
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

SKIP_DIRS = {"@eaDir", "#recycle", ".git"}

THUMB_CACHE_DIR = STORAGE_DIR / ".thumbcache"
BACKUP_ROOT = STORAGE_DIR / "backups"

app = FastAPI(title="Photodate")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# In-memory analysis job state (resets on restart; persisted status in AlbumData)
_analysis_jobs: dict[str, dict] = {}
_collection_jobs: dict[str, dict] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_photos_roots() -> list[Path]:
    raw = os.environ.get("PHOTOS_PATHS", os.environ.get("PHOTOS_PATH", "/photos"))
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def _album_status_label(album_data: AlbumData) -> str:
    if album_data.applied_count > 0:
        return "toegepast"
    if album_data.analysis_status == "done" and album_data.photo_dates:
        return "geanalyseerd"
    if album_data.analysis_status == "running":
        return "bezig"
    if album_data.analysis_status == "error":
        return "fout"
    return "nieuw"


def _get_all_albums() -> list[tuple[Path, str, str]]:
    """Return (folder, rel_label, status) for every album under all roots."""
    albums = []
    for root in _get_photos_roots():
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            dirpath = Path(dirpath)
            has_photos = any(Path(f).suffix.lower() in EXTENSIONS for f in filenames)
            if has_photos and dirpath != root:
                rel = str(dirpath.relative_to(root))
                album_data = AlbumData.load(rel)
                # Check live job state too
                job = _analysis_jobs.get(rel)
                if job and job["status"] == "running":
                    status = "bezig"
                else:
                    status = _album_status_label(album_data)
                albums.append((dirpath, rel, status))
    albums.sort(key=lambda x: x[1])
    return albums


def _find_album(album_rel: str) -> Path | None:
    for root in _get_photos_roots():
        candidate = root / album_rel
        if candidate.is_dir():
            return candidate
    return None


def _load_photos(folder: Path) -> list[PhotoInfo]:
    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    )
    return [
        PhotoInfo(path=f, index=i, original_exif_date=read_exif_date(f))
        for i, f in enumerate(files)
    ]


# ── Thumbnail with disk cache ─────────────────────────────────────────────────

def _thumbnail_bytes(folder: Path, album_rel: str, filename: str, size: int) -> bytes | None:
    source = folder / filename
    if not source.is_file():
        return None

    safe_rel = album_rel.replace("/", "__").replace("\\", "__")
    cache_path = THUMB_CACHE_DIR / safe_rel / str(size) / (filename + ".jpg")

    try:
        if (
            cache_path.exists()
            and cache_path.stat().st_mtime >= source.stat().st_mtime
        ):
            return cache_path.read_bytes()
    except OSError:
        pass

    img = Image.open(source)
    img.thumbnail((size, size))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    data = buf.getvalue()

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    except OSError:
        pass

    return data


@app.get("/thumb/{album_rel:path}/__file__/{filename}")
async def thumbnail(album_rel: str, filename: str, size: int = 150):
    folder = _find_album(album_rel)
    if not folder:
        return Response(status_code=404)
    data = _thumbnail_bytes(folder, album_rel, filename, size)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


# ── Async analysis ────────────────────────────────────────────────────────────

async def _run_analysis_task(album_rel: str, skip_faces: bool = False) -> None:
    job = _analysis_jobs[album_rel]
    album_data_ref: AlbumData | None = None
    try:
        folder = _find_album(album_rel)
        if not folder:
            raise ValueError(f"Album niet gevonden: {album_rel}")

        photos = _load_photos(folder)
        album_data_ref = AlbumData.load(album_rel)
        settings = GlobalSettings.load()

        # Step 1: Face detection (CPU-bound → thread pool)
        if not skip_faces:
            job.update({"step": "faces", "progress": 0, "total": len(photos)})
            try:
                face_results = await asyncio.to_thread(
                    detect_faces_in_album,
                    [p.path for p in photos],
                    [{"name": fp.name, "encoding": fp.encoding} for fp in settings.face_profiles],
                )
            except Exception as exc:
                logger.warning("Gezichtsdetectie overgeslagen: %s", exc)
                face_results = {"clusters": [], "per_photo": {}}

            album_data_ref.face_clusters = face_results["clusters"]
            album_data_ref.faces_per_photo = {
                str(k): v for k, v in face_results.get("per_photo", {}).items()
            }
            # Auto-apply labels from global profiles
            for cluster in album_data_ref.face_clusters:
                auto = cluster.get("auto_label", "")
                if auto and cluster["id"] not in album_data_ref.face_labels:
                    album_data_ref.face_labels[cluster["id"]] = auto

        # Step 2: AI date analysis (sync OpenAI SDK → thread pool)
        job.update({"step": "ai", "progress": 0, "total": len(photos)})

        def _progress(step: str, current: int, total: int) -> None:
            job.update({"step": step, "progress": current, "total": total})

        client = openai.OpenAI()
        album_data_ref = await asyncio.to_thread(
            analyze_album_full,
            photos,
            album_data_ref,
            settings,
            client,
            _progress,
        )

        job["status"] = "done"
        logger.info("Analyse klaar: %s", album_rel)

    except Exception as exc:
        logger.error("Analyse mislukt voor %s: %s\n%s", album_rel, exc, traceback.format_exc())
        job.update({"status": "error", "error": str(exc)})
        if album_data_ref is not None:
            album_data_ref.analysis_status = "error"
            album_data_ref.analysis_error = str(exc)
            album_data_ref.save()


@app.post("/album/{album_rel:path}/analyze", response_class=HTMLResponse)
async def album_analyze(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    _analysis_jobs[album_rel] = {
        "status": "running",
        "step": "start",
        "progress": 0,
        "total": 0,
        "error": "",
    }

    # Mark as running in persistent storage so home page shows correct status
    album_data = AlbumData.load(album_rel)
    album_data.analysis_status = "running"
    album_data.analysis_error = ""
    album_data.save()

    asyncio.create_task(_run_analysis_task(album_rel, skip_faces=False))

    photos = _load_photos(folder)
    return templates.TemplateResponse(request, "progress.html", {
        "request": request,
        "album_rel": album_rel,
        "folder_name": folder.name,
        "total": len(photos),
        "reanalyze": False,
    })


@app.post("/album/{album_rel:path}/reanalyze", response_class=HTMLResponse)
async def album_reanalyze(request: Request, album_rel: str):
    """Re-run AI analysis only (keeps existing face labels, skips face detection)."""
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    _analysis_jobs[album_rel] = {
        "status": "running",
        "step": "ai",
        "progress": 0,
        "total": 0,
        "error": "",
    }

    album_data = AlbumData.load(album_rel)
    album_data.analysis_status = "running"
    album_data.analysis_error = ""
    album_data.save()

    asyncio.create_task(_run_analysis_task(album_rel, skip_faces=True))

    photos = _load_photos(folder)
    return templates.TemplateResponse(request, "progress.html", {
        "request": request,
        "album_rel": album_rel,
        "folder_name": folder.name,
        "total": len(photos),
        "reanalyze": True,
    })


@app.get("/album/{album_rel:path}/progress", response_class=HTMLResponse)
async def album_progress_page(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)
    photos = _load_photos(folder)
    return templates.TemplateResponse(request, "progress.html", {
        "request": request,
        "album_rel": album_rel,
        "folder_name": folder.name,
        "total": len(photos),
        "reanalyze": False,
    })


@app.get("/album/{album_rel:path}/status")
async def album_status_endpoint(album_rel: str):
    job = _analysis_jobs.get(album_rel)
    if job is not None:
        return JSONResponse(job)

    album_data = AlbumData.load(album_rel)
    status = album_data.analysis_status
    # Running in storage but not in memory = container restarted mid-analysis
    if status == "running":
        status = "lost"
    return JSONResponse({
        "status": status,
        "step": "",
        "progress": 0,
        "total": 0,
        "error": album_data.analysis_error,
    })


# ── Page: Results (after analysis) ───────────────────────────────────────────

@app.get("/album/{album_rel:path}/results", response_class=HTMLResponse)
async def album_results_page(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)
    album_data = AlbumData.load(album_rel)
    photos = _load_photos(folder)
    # Convert string keys back to int for template iteration by photo.index
    faces_per_photo = {int(k): v for k, v in album_data.faces_per_photo.items()}
    return templates.TemplateResponse(request, "results.html", {
        "request": request,
        "album_rel": album_rel,
        "photos": photos,
        "album_data": album_data,
        "face_clusters": album_data.face_clusters,
        "faces_per_photo": faces_per_photo,
    })


# ── Page: Home ────────────────────────────────────────────────────────────────

def _collections(folders: list[tuple[Path, str, str]]) -> list[tuple[str, int]]:
    """Parent folders holding two or more albums."""
    counts: dict[str, int] = {}
    for _, rel, _ in folders:
        parent = str(Path(rel).parent)
        if parent != ".":
            counts[parent] = counts.get(parent, 0) + 1
    return sorted((p, n) for p, n in counts.items() if n >= 2)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    folders = _get_all_albums()
    roots = _get_photos_roots()
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "folders": folders,
        "collections": _collections(folders),
        "photos_roots": roots,
    })


# ── Page: Collection (date many albums at once, no AI) ────────────────────────

def _load_collection(rel: str) -> list[coll.CollectionAlbum] | None:
    folder = _find_album(rel)
    if not folder:
        return None
    manual = {}
    for sub in folder.iterdir():
        if sub.is_dir():
            sub_rel = f"{rel}/{sub.name}"
            data = AlbumData.load(sub_rel)
            if data.start_date or data.end_date:
                manual[sub_rel] = (data.start_date, data.end_date)
    return coll.load_collection(folder, rel, manual)


def _run_collection_job(rel: str, targets: list[coll.CollectionAlbum]) -> None:
    job = _collection_jobs[rel]
    try:
        for album in targets:
            job["current"] = album.name
            written, errors = coll.apply_album(album, coll.backup_dir_for(BACKUP_ROOT, album.rel))
            data = AlbumData.load(album.rel)
            for filename, iso in written.items():
                data.photo_dates[filename] = iso[:10]
                data.photo_date_source[filename] = "spreiding"
            data.applied_count = len(written)
            data.save()
            job["photos"] += len(written)
            job["albums_done"] += 1
            job["errors"].extend(f"{album.name} – {e}" for e in errors)
        job["status"] = "done"
    except Exception as exc:
        logger.error("Collectie dateren mislukt: %s", traceback.format_exc())
        job["status"] = "error"
        job["errors"].append(str(exc))
    job["current"] = ""


@app.get("/collection/{rel:path}", response_class=HTMLResponse)
async def collection_page(request: Request, rel: str, saved: bool = False):
    albums = await asyncio.to_thread(_load_collection, rel)
    if albums is None:
        return HTMLResponse("Map niet gevonden", status_code=404)
    rows = []
    for a in albums:
        plan = a.planned()
        samples = [plan[0], plan[len(plan) // 2], plan[-1]] if plan else []
        data = AlbumData.load(a.rel)
        manual_start, manual_end = data.start_date[:7], data.end_date[:7]
        rows.append({
            "album": a,
            "samples": [(p.name, dt) for p, dt in samples],
            "dated": sum(1 for s in data.photo_date_source.values() if s == "spreiding"),
            "manual_start": manual_start,
            "manual_end": manual_end,
            "has_backup": coll.backup_dir_for(BACKUP_ROOT, a.rel).is_dir(),
        })
    ready = [r for r in rows if r["album"].complete]
    return templates.TemplateResponse(request, "collection.html", {
        "request": request,
        "rel": rel,
        "rows": rows,
        "ready_albums": len(ready),
        "ready_photos": sum(len(r["album"].photos) for r in ready),
        "job": _collection_jobs.get(rel),
        "saved": saved,
        "period_label": coll.period_label,
    })


@app.post("/collection/{rel:path}/periods")
async def collection_save_periods(request: Request, rel: str):
    form = await request.form()
    i = 0
    while f"album_{i}" in form:
        album_rel = form[f"album_{i}"]
        start = coll.parse_manual_date(form.get(f"start_{i}", ""), is_end=False)
        end = coll.parse_manual_date(form.get(f"end_{i}", ""), is_end=True)
        data = AlbumData.load(album_rel)
        data.start_date = start.isoformat() if start else ""
        data.end_date = end.isoformat() if end else ""
        data.save()
        i += 1
    return RedirectResponse(f"/collection/{rel}?saved=1", status_code=303)


@app.post("/collection/{rel:path}/apply")
async def collection_apply(request: Request, rel: str):
    form = await request.form()
    target = form.get("album", "")
    job = _collection_jobs.get(rel)
    if job and job["status"] == "running":
        return RedirectResponse(f"/collection/{rel}", status_code=303)
    albums = await asyncio.to_thread(_load_collection, rel) or []
    targets = [a for a in albums if a.complete and (target == "__all__" or a.rel == target)]
    _collection_jobs[rel] = {
        "status": "running", "current": "", "albums_total": len(targets),
        "albums_done": 0, "photos": 0, "errors": [],
    }
    asyncio.create_task(asyncio.to_thread(_run_collection_job, rel, targets))
    return RedirectResponse(f"/collection/{rel}", status_code=303)


@app.post("/collection/{rel:path}/restore")
async def collection_restore(request: Request, rel: str):
    form = await request.form()
    album_rel = form.get("album", "")
    folder = _find_album(album_rel)
    if folder:
        await asyncio.to_thread(
            coll.restore_album, folder, coll.backup_dir_for(BACKUP_ROOT, album_rel)
        )
        data = AlbumData.load(album_rel)
        for filename in [f for f, s in data.photo_date_source.items() if s == "spreiding"]:
            data.photo_date_source.pop(filename, None)
            data.photo_dates.pop(filename, None)
        data.applied_count = 0
        data.save()
    return RedirectResponse(f"/collection/{rel}", status_code=303)


# ── Page: Global Settings ─────────────────────────────────────────────────────

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    settings = GlobalSettings.load()
    return templates.TemplateResponse(request, "settings.html", {
        "request": request,
        "settings": settings,
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

    settings = GlobalSettings.load()
    settings.family_members = members
    settings.save()
    return templates.TemplateResponse(request, "settings.html", {
        "request": request,
        "settings": settings,
        "saved": True,
    })


@app.post("/settings/delete-face-profile/{profile_name}", response_class=HTMLResponse)
async def delete_face_profile(request: Request, profile_name: str):
    settings = GlobalSettings.load()
    settings.face_profiles = [p for p in settings.face_profiles if p.name != profile_name]
    settings.save()
    return RedirectResponse("/settings", status_code=303)


# ── Page: Album context ───────────────────────────────────────────────────────

@app.get("/album/{album_rel:path}", response_class=HTMLResponse)
async def album_context_page(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)
    album_data = AlbumData.load(album_rel)
    photos = _load_photos(folder)
    photo_filenames_json = json.dumps([p.path.name for p in photos])
    return templates.TemplateResponse(request, "album_context.html", {
        "request": request,
        "album_rel": album_rel,
        "album_data": album_data,
        "photo_count": len(photos),
        "photo_filenames_json": photo_filenames_json,
        "folder_name": folder.name,
    })


@app.post("/album/{album_rel:path}/save", response_class=HTMLResponse)
async def album_context_save(request: Request, album_rel: str):
    form = await request.form()
    album_data = AlbumData.load(album_rel)
    album_data.start_date = form.get("start_date", "").strip()
    album_data.end_date = form.get("end_date", "").strip()
    album_data.context_notes = form.get("context_notes", "").strip()

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
    photo_filenames_json = json.dumps([p.path.name for p in photos])
    return templates.TemplateResponse(request, "album_context.html", {
        "request": request,
        "album_rel": album_rel,
        "album_data": album_data,
        "photo_count": len(photos),
        "photo_filenames_json": photo_filenames_json,
        "folder_name": folder.name,
        "saved": True,
    })


# ── Page: Save face labels ────────────────────────────────────────────────────

@app.post("/album/{album_rel:path}/faces", response_class=HTMLResponse)
async def save_face_labels(request: Request, album_rel: str):
    form = await request.form()
    album_data = AlbumData.load(album_rel)
    settings = GlobalSettings.load()
    settings_changed = False

    for key, value in form.items():
        if key.startswith("face_label_") and value.strip():
            face_id = key[len("face_label_"):]
            album_data.face_labels[face_id] = value.strip()

            # Optionally save as global face profile
            if form.get(f"face_save_global_{face_id}") == "1":
                centroid = None
                for cluster in album_data.face_clusters:
                    if cluster.get("id") == face_id and "centroid" in cluster:
                        centroid = cluster["centroid"]
                        break
                if centroid is not None:
                    name = value.strip()
                    existing = next(
                        (p for p in settings.face_profiles if p.name == name), None
                    )
                    if existing:
                        existing.encoding = centroid
                    else:
                        settings.face_profiles.append(
                            FaceProfile(name=name, encoding=centroid)
                        )
                    settings_changed = True

    album_data.save()
    if settings_changed:
        settings.save()

    folder = _find_album(album_rel)
    photos = _load_photos(folder)
    return templates.TemplateResponse(request, "dates.html", {
        "request": request,
        "album_rel": album_rel,
        "photos": photos,
        "album_data": album_data,
    })


# ── Page: Date review ─────────────────────────────────────────────────────────

@app.get("/album/{album_rel:path}/dates", response_class=HTMLResponse)
async def date_review_page(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)
    photos = _load_photos(folder)
    album_data = AlbumData.load(album_rel)
    return templates.TemplateResponse(request, "dates.html", {
        "request": request,
        "album_rel": album_rel,
        "photos": photos,
        "album_data": album_data,
    })


# ── Page: Apply dates ─────────────────────────────────────────────────────────

@app.post("/album/{album_rel:path}/apply", response_class=HTMLResponse)
async def apply_dates(request: Request, album_rel: str):
    form = await request.form()
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    album_data = AlbumData.load(album_rel)

    # Persist "niet van mij" marks
    not_mine = [
        key[len("not_mine_"):]
        for key, value in form.items()
        if key.startswith("not_mine_") and value == "1"
    ]
    album_data.not_mine_photos = not_mine
    album_data.save()

    # Copy not-mine photos to {root}/nietzelfgenomen/{album_rel}/
    copied = 0
    if not_mine:
        root_folder = _find_album(album_rel)
        for photos_root in _get_photos_roots():
            if root_folder and str(root_folder).startswith(str(photos_root)):
                dest_dir = photos_root / "nietzelfgenomen" / album_rel
                dest_dir.mkdir(parents=True, exist_ok=True)
                for fname in not_mine:
                    src = folder / fname
                    if src.is_file():
                        dest = dest_dir / fname
                        if not dest.exists():
                            shutil.copy2(src, dest)
                            copied += 1
                break

    count = 0
    skipped = 0
    errors = 0
    for key, value in form.items():
        if key.startswith("date_") and value:
            filename = key[5:]
            if filename in not_mine:
                skipped += 1
                continue
            filepath = folder / filename
            if filepath.is_file():
                try:
                    dt = date.fromisoformat(value)
                    write_exif_date(filepath, dt, backup=True)
                    count += 1
                except ValueError:
                    logger.warning("Ongeldige datum voor %s: %s", filename, value)
                    errors += 1
                except Exception as exc:
                    logger.warning("Kon datum niet schrijven voor %s: %s", filename, exc)
                    errors += 1

    album_data.applied_count = count
    album_data.save()

    return templates.TemplateResponse(request, "applied.html", {
        "request": request,
        "album_rel": album_rel,
        "folder_name": folder.name,
        "count": count,
        "skipped": skipped,
        "copied": copied,
        "errors": errors,
    })


# ── Page: Backup restore ──────────────────────────────────────────────────────

@app.get("/album/{album_rel:path}/backups", response_class=HTMLResponse)
async def backups_page(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)
    photos = _load_photos(folder)
    backups = [
        p.path.name
        for p in photos
        if p.path.with_suffix(p.path.suffix + ".bak").exists()
    ]
    return templates.TemplateResponse(request, "backups.html", {
        "request": request,
        "album_rel": album_rel,
        "folder_name": folder.name,
        "backups": backups,
    })


@app.post("/album/{album_rel:path}/restore")
async def restore_backups(request: Request, album_rel: str):
    form = await request.form()
    folder = _find_album(album_rel)
    if not folder:
        return JSONResponse({"error": "Map niet gevonden"}, status_code=404)

    restored = 0
    errors = []
    for key, value in form.items():
        if key.startswith("restore_") and value == "1":
            filename = key[len("restore_"):]
            filepath = folder / filename
            backup = filepath.with_suffix(filepath.suffix + ".bak")
            if backup.exists():
                try:
                    shutil.copy2(backup, filepath)
                    restored += 1
                except Exception as exc:
                    errors.append(f"{filename}: {exc}")
            else:
                errors.append(f"{filename}: geen backup gevonden")

    photos = _load_photos(folder)
    backups = [
        p.path.name
        for p in photos
        if p.path.with_suffix(p.path.suffix + ".bak").exists()
    ]
    return templates.TemplateResponse(request, "backups.html", {
        "request": request,
        "album_rel": album_rel,
        "folder_name": folder.name,
        "backups": backups,
        "restored": restored,
        "restore_errors": errors,
    })

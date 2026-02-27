import os
from datetime import date
from io import BytesIO
from pathlib import Path

import openai
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from PIL import Image

from .analyzer import analyze_album_full
from .exif import read_exif_date, write_exif_date
from .faces import detect_faces_in_album
from .models import PhotoInfo
from .storage import AlbumData, FamilyMember, GlobalSettings, Milestone

app = FastAPI(title="Photodate")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
SKIP_DIRS = {"@eaDir", "#recycle", ".git"}


# --- Helpers ---

def _get_photos_roots() -> list[Path]:
    raw = os.environ.get("PHOTOS_PATHS", os.environ.get("PHOTOS_PATH", "/photos"))
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def _get_all_albums() -> list[tuple[Path, str]]:
    albums = []
    for root in _get_photos_roots():
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            dirpath = Path(dirpath)
            has_photos = any(Path(f).suffix.lower() in EXTENSIONS for f in filenames)
            if has_photos and dirpath != root:
                rel = dirpath.relative_to(root)
                albums.append((dirpath, str(rel)))
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
    folders = _get_all_albums()
    roots = _get_photos_roots()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "folders": folders,
        "photos_roots": roots,
    })


# --- Page: Global Settings (family members) ---

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    settings = GlobalSettings.load()
    return templates.TemplateResponse("settings.html", {
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
    settings = GlobalSettings(family_members=members)
    settings.save()
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "settings": settings,
        "saved": True,
    })


# --- Page: Album context (step 1) ---

@app.get("/album/{album_rel:path}", response_class=HTMLResponse)
async def album_context_page(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)
    album_data = AlbumData.load(album_rel)
    photos = _load_photos(folder)
    return templates.TemplateResponse("album_context.html", {
        "request": request,
        "album_rel": album_rel,
        "album_data": album_data,
        "photo_count": len(photos),
        "folder_name": folder.name,
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

    return templates.TemplateResponse("album_context.html", {
        "request": request,
        "album_rel": album_rel,
        "album_data": album_data,
        "photo_count": len(_load_photos(_find_album(album_rel))),
        "folder_name": _find_album(album_rel).name,
        "saved": True,
    })


# --- Page: Analyze (step 2: face detection + AI analysis) ---

@app.post("/album/{album_rel:path}/analyze", response_class=HTMLResponse)
async def album_analyze(request: Request, album_rel: str):
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    photos = _load_photos(folder)
    album_data = AlbumData.load(album_rel)
    settings = GlobalSettings.load()

    # Face detection
    face_results = detect_faces_in_album([p.path for p in photos])

    # AI date analysis
    try:
        client = openai.OpenAI()
        album_data = analyze_album_full(photos, album_data, settings, client)
    except Exception as e:
        return HTMLResponse(f"<h2>Fout bij AI analyse</h2><pre>{e}</pre>", status_code=500)

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


# --- Page: Date review (step 3) ---

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


# --- Page: Apply dates (step 4) ---

@app.post("/album/{album_rel:path}/apply", response_class=HTMLResponse)
async def apply_dates(request: Request, album_rel: str):
    form = await request.form()
    folder = _find_album(album_rel)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    count = 0
    for key, value in form.items():
        if key.startswith("date_") and value:
            filename = key[5:]
            filepath = folder / filename
            if filepath.is_file():
                try:
                    dt = date.fromisoformat(value)
                    write_exif_date(filepath, dt, backup=True)
                    count += 1
                except (ValueError, Exception):
                    pass

    return templates.TemplateResponse("applied.html", {
        "request": request,
        "album_rel": album_rel,
        "folder_name": folder.name,
        "count": count,
    })

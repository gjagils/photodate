import os
from datetime import date
from pathlib import Path

import openai
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .analyzer import analyze_album
from .exif import read_exif_date, write_exif_date
from .models import PhotoInfo
from .validator import scan_and_validate

app = FastAPI(title="Photodate")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

def _get_photos_roots() -> list[Path]:
    """Parse PHOTOS_PATHS env var into a list of Paths."""
    raw = os.environ.get("PHOTOS_PATHS", os.environ.get("PHOTOS_PATH", "/photos"))
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def _get_all_albums() -> list[tuple[Path, str]]:
    """Recursively find all folders containing images. Returns (path, relative_label)."""
    extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    albums = []
    for root in _get_photos_roots():
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirpath = Path(dirpath)
            has_photos = any(
                Path(f).suffix.lower() in extensions for f in filenames
            )
            if has_photos:
                rel = dirpath.relative_to(root)
                albums.append((dirpath, str(rel)))
    albums.sort(key=lambda x: x[1])
    return albums


def _find_album(album_rel: str) -> Path | None:
    """Find an album folder by relative path across all photo roots."""
    for root in _get_photos_roots():
        candidate = root / album_rel
        if candidate.is_dir():
            return candidate
    return None


def _load_photos(folder: Path) -> list[PhotoInfo]:
    extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )
    return [
        PhotoInfo(path=f, index=i, original_exif_date=read_exif_date(f))
        for i, f in enumerate(files)
    ]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    folders = _get_all_albums()
    roots = _get_photos_roots()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "folders": folders,
        "photos_roots": roots,
    })


@app.get("/validate/{folder_name:path}", response_class=HTMLResponse)
async def validate_folder(request: Request, folder_name: str):
    folder = _find_album(folder_name)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    photos = scan_and_validate(folder)
    return templates.TemplateResponse("validate.html", {
        "request": request,
        "folder_name": folder_name,
        "photos": photos,
    })


@app.post("/analyze/{folder_name:path}", response_class=HTMLResponse)
async def analyze_folder(request: Request, folder_name: str, context: str = Form(...)):
    folder = _find_album(folder_name)
    if not folder:
        return HTMLResponse("Map niet gevonden", status_code=404)

    photos = _load_photos(folder)
    client = openai.OpenAI()
    photos = analyze_album(photos, context, client)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "folder_name": folder_name,
        "context": context,
        "photos": photos,
    })


@app.post("/apply", response_class=HTMLResponse)
async def apply_dates(request: Request):
    form = await request.form()
    folder_name = form.get("folder_name")
    count = 0

    for key, value in form.items():
        if key.startswith("date_"):
            filepath = form.get(f"path_{key[5:]}")
            if filepath and value:
                try:
                    dt = date.fromisoformat(value)
                    write_exif_date(Path(filepath), dt, backup=True)
                    count += 1
                except (ValueError, Exception):
                    pass

    return templates.TemplateResponse("applied.html", {
        "request": request,
        "folder_name": folder_name,
        "count": count,
    })

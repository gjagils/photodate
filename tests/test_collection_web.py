"""Collection page end-to-end through the FastAPI app."""
import time

import piexif
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    for name in ["IX - Okt '88 tot Jan '90", "X - Jan '90 tot Jul '90", "XII"]:
        folder = photos / "Gils" / name
        folder.mkdir(parents=True)
        for i in range(1, 4):
            Image.new("RGB", (40, 30), (20 * i, 40, 60)).save(folder / f"boek_{i:03d}.jpg")
    monkeypatch.setenv("PHOTOS_PATHS", str(photos))
    from photodate import web
    import photodate.storage as storage
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "data")
    monkeypatch.setattr(web, "BACKUP_ROOT", tmp_path / "data" / "backups")
    web._collection_jobs.clear()
    return TestClient(web.app), photos


def _wait_done(web_client, rel):
    from photodate import web
    for _ in range(100):
        if web._collection_jobs.get(rel, {}).get("status") != "running":
            return
        time.sleep(0.05)


def test_index_lists_collection(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "/collection/Gils" in r.text


def test_date_folders_are_not_collections():
    from photodate.web import _collections
    albums = [{"label": l} for l in [
        "Fotobibliotheek/2001/01", "Fotobibliotheek/2001/02",
        "MobileBackup/iPhone/2014", "MobileBackup/iPhone/2015",
        "Gils/IX - Okt '88 tot Jan '90", "Gils/XII",
    ]]
    assert _collections(albums) == [("Gils", 2)]


def test_collection_page_shows_periods(client):
    c, _ = client
    r = c.get("/collection/Gils")
    assert r.status_code == 200
    assert "okt 1988 – jan 1990" in r.text
    assert "ontbreekt" in r.text  # XII has no next neighbour for its end


def test_save_period_and_apply_single_album(client):
    c, photos = client
    r = c.post("/collection/Gils/periods", data={
        "album_0": "Gils/IX - Okt '88 tot Jan '90", "start_0": "", "end_0": "",
        "album_1": "Gils/X - Jan '90 tot Jul '90", "start_1": "", "end_1": "",
        "album_2": "Gils/XII", "start_2": "", "end_2": "1992-11",
    }, follow_redirects=False)
    assert r.status_code == 303

    r = c.post("/collection/Gils/apply", data={"album": "Gils/XII"}, follow_redirects=False)
    assert r.status_code == 303
    _wait_done(c, "Gils")

    dated = piexif.load(str(photos / "Gils" / "XII" / "boek_003.jpg"))
    assert dated["Exif"][piexif.ExifIFD.DateTimeOriginal].startswith(b"1992:11:30")
    untouched = piexif.load(str(photos / "Gils" / "IX - Okt '88 tot Jan '90" / "boek_001.jpg"))
    assert piexif.ExifIFD.DateTimeOriginal not in untouched.get("Exif", {})

    page = c.get("/collection/Gils").text
    assert "3 gedateerd" in page
    assert "Herstel origineel" in page

    c.post("/collection/Gils/restore", data={"album": "Gils/XII"})
    restored = piexif.load(str(photos / "Gils" / "XII" / "boek_003.jpg"))
    assert piexif.ExifIFD.DateTimeOriginal not in restored.get("Exif", {})

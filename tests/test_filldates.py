"""Tests for filldates.py: date sources for library photos without EXIF date."""
import json
import os
from datetime import datetime
from pathlib import Path

import piexif
from PIL import Image

from photodate import filldates as fd


def _jpg(path: Path, date: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (30, 20), (10, 120, 200)).save(path)
    if date:
        piexif.insert(piexif.dump({"Exif": {piexif.ExifIFD.DateTimeOriginal: date.encode()}}), str(path))
    return path


def _sidecar(path: Path, ts: int) -> Path:
    path.write_text(json.dumps({"photoTakenTime": {"timestamp": str(ts)}}))
    return path


def test_name_time_patterns():
    assert fd.name_time("IMG_20190612_153012.jpg") == datetime(2019, 6, 12, 15, 30, 12)
    assert fd.name_time("2018-02-03_7f3a.JPG") == datetime(2018, 2, 3, 12, 0)
    assert fd.name_time("vakantie_2025-05-11_ab12.jpg") == datetime(2025, 5, 11, 12, 0)
    assert fd.name_time("IMG-20190612-WA0001.jpg") == datetime(2019, 6, 12, 12, 0)
    assert fd.name_time("Screenshot_2019-06-12-15-30-12.png") == datetime(2019, 6, 12, 15, 30, 12)
    assert fd.name_time("01bb8371-968b-4ae5-9f10-e6fab2c70aef(1).jpg") is None
    assert fd.name_time("IMG_0819.JPG") is None


def test_folder_month():
    assert fd.folder_month(Path("Fotobibliotheek/2019/06/x.jpg")) == (2019, 6)
    assert fd.folder_month(Path("Fotobibliotheek/2019/00/x.jpg")) == (2019, None)
    assert fd.folder_month(Path("Fotobibliotheek/2019/x.jpg")) == (2019, None)
    assert fd.folder_month(Path("Uit te zoeken/x.jpg")) == (None, None)


def test_sidecar_variants(tmp_path):
    names = ["IMG_2557.HEIC.json", "abc.jpg(1).json", "1c5c9811-24e0.jpg.json",
             "a_very_long_photo_name_from_takeout_export.jpg.supplemen.json"]
    jsons = {n.lower(): tmp_path / n for n in names}
    assert fd.sidecar_for(Path("IMG_2557.HEIC"), jsons).name == "IMG_2557.HEIC.json"
    assert fd.sidecar_for(Path("abc(1).jpg"), jsons).name == "abc.jpg(1).json"
    assert fd.sidecar_for(Path("1c5c9811-24e0-bewerkt.jpg"), jsons).name == "1c5c9811-24e0.jpg.json"
    assert fd.sidecar_for(Path("a_very_long_photo_name_from_takeout_export.jpg"), jsons) is not None
    assert fd.sidecar_for(Path("other.jpg"), jsons) is None


def test_plan_folder_sources_and_skip_dated(tmp_path):
    month = tmp_path / "2019" / "06"
    _jpg(month / "dated.jpg", "2019:06:01 10:00:00")
    _sidecar(month / "take.jpg.json", 1560350000)  # 2019-06-12 16:33 Amsterdam
    _jpg(month / "take.jpg")
    _jpg(month / "IMG_20190620_101500.jpg")
    old = _jpg(month / "mtime.jpg")
    ts = datetime(2019, 6, 25, 9, 0).timestamp()
    os.utime(old, (ts, ts))
    _jpg(month / "a.jpg")
    _jpg(month / "b.jpg")
    _jpg(tmp_path / "2019" / "00" / "unknown.jpg")

    plans = {p.path.name: p for p in fd.plan_folder(tmp_path / "2019", workers=2)}
    assert "dated.jpg" not in plans
    assert plans["take.jpg"].source == "takeout"
    assert plans["take.jpg"].when == datetime(2019, 6, 12, 16, 33, 20)
    assert plans["IMG_20190620_101500.jpg"].source == "bestandsnaam"
    assert plans["mtime.jpg"].source == "bestandsdatum"
    assert plans["a.jpg"].source == plans["b.jpg"].source == "map"
    assert plans["a.jpg"].when == datetime(2019, 6, 15, 12, 0)
    assert plans["b.jpg"].when == datetime(2019, 6, 15, 12, 1)
    assert plans["unknown.jpg"].note == "2019, maand onbekend"
    assert plans["unknown.jpg"].source == "jaar"


def test_write_plan_sets_date_description_and_backup(tmp_path):
    photo = _jpg(tmp_path / "2019" / "06" / "x.jpg")
    plan = fd.Plan(photo, datetime(2019, 6, 15, 12, 0), "map", "juni 2019")
    fd.write_plan(plan, tmp_path / "backups")
    exif = piexif.load(str(photo))
    assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2019:06:15 12:00:00"
    assert exif["0th"][piexif.ImageIFD.ImageDescription] == b"Datum geschat: juni 2019 (uit map)"
    assert datetime.fromtimestamp(photo.stat().st_mtime) == datetime(2019, 6, 15, 12, 0)
    assert (tmp_path / "backups" / "06" / "x.jpg").exists()
    assert fd.exif_date(photo) == "2019:06:15 12:00:00"


def test_copies_borrow_the_date_of_their_original(tmp_path):
    _jpg(tmp_path / "2019" / "05" / "abc.jpg", "2019:05:04 10:00:00")
    _jpg(tmp_path / "2019" / "08" / "def.jpg")
    _sidecar(tmp_path / "2019" / "08" / "def.jpg.json", 1565000000)
    _jpg(tmp_path / "2019" / "00" / "abc(1).jpg")
    _jpg(tmp_path / "2019" / "00" / "def-bewerkt.jpg")
    _jpg(tmp_path / "2019" / "00" / "lonely(1).jpg")
    plans = {p.path.name: p for p in fd.plan_folder(tmp_path / "2019", workers=2)}
    assert plans["abc(1).jpg"].source == "origineel"
    assert plans["abc(1).jpg"].when == datetime(2019, 5, 4, 10, 0)
    assert plans["def-bewerkt.jpg"].source == "origineel"
    assert plans["def-bewerkt.jpg"].when == plans["def.jpg"].when
    assert plans["lonely(1).jpg"].source == "jaar"


def test_original_name():
    assert fd.original_name("IMG_1(1).HEIC") == "IMG_1.HEIC"
    assert fd.original_name("x-bewerkt.jpg") == "x.jpg"
    assert fd.original_name("IMG_1.HEIC") is None


def test_numbered_downloads_are_not_copies(tmp_path):
    month = tmp_path / "2019" / "05"
    month.mkdir(parents=True)
    Image.new("RGB", (525, 660)).save(month / "cover.jpg")
    piexif.insert(piexif.dump({"Exif": {piexif.ExifIFD.DateTimeOriginal: b"2019:05:22 14:52:57"}}), str(month / "cover.jpg"))
    other = tmp_path / "2019" / "00"
    other.mkdir(parents=True)
    Image.new("RGB", (1275, 1651)).save(other / "cover(126).jpg")   # different picture
    Image.new("RGB", (262, 330)).save(other / "cover(1).jpg")       # resized copy, same ratio
    plans = {p.path.name: p for p in fd.plan_folder(tmp_path / "2019", workers=2)}
    assert plans["cover(126).jpg"].source == "jaar"
    assert plans["cover(1).jpg"].source == "origineel"


def test_settle_mtimes(tmp_path):
    photo = _jpg(tmp_path / "x.jpg")
    plan = fd.Plan(photo, datetime(2019, 6, 15, 12, 0), "map", "juni 2019")
    assert fd.settle_mtimes([plan], wait=0) == 1
    assert datetime.fromtimestamp(photo.stat().st_mtime) == datetime(2019, 6, 15, 12, 0)
    assert fd.settle_mtimes([plan], wait=0) == 0


import hashlib
import shutil as _shutil

import pytest


def _pixels(path: Path) -> str:
    with Image.open(path) as im:
        return hashlib.md5(im.convert("RGB").tobytes()).hexdigest()


@pytest.mark.skipif(_shutil.which("exiftool") is None, reason="exiftool niet geïnstalleerd")
@pytest.mark.parametrize("ext", ["png", "heic"])
def test_exiftool_writes_date_without_touching_pixels(tmp_path, ext):
    if ext == "heic":
        pytest.importorskip("pillow_heif")
    photo = tmp_path / "2019" / "06" / f"IMG_0001.{ext}"
    photo.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48), (200, 40, 90)).save(photo)
    before = _pixels(photo)
    plan = fd.Plan(photo, datetime(2019, 6, 12, 16, 33, 20), "takeout")
    fd.write_plan(plan, tmp_path / "backups")
    assert fd.exif_date(photo) == "2019:06:12 16:33:20"
    assert _pixels(photo) == before
    assert (tmp_path / "backups" / "06" / photo.name).exists()
    assert datetime.fromtimestamp(photo.stat().st_mtime) == datetime(2019, 6, 12, 16, 33, 20)


def test_can_write(monkeypatch):
    monkeypatch.setattr(fd, "EXIFTOOL", None)
    assert fd.can_write(Path("a.JPG")) and not fd.can_write(Path("a.HEIC"))
    monkeypatch.setattr(fd, "EXIFTOOL", "/usr/bin/exiftool")
    assert fd.can_write(Path("a.HEIC")) and fd.can_write(Path("a.png")) and not fd.can_write(Path("a.gif"))

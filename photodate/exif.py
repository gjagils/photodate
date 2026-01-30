import shutil
from datetime import date, datetime
from pathlib import Path

import piexif


def read_exif_date(path: Path) -> str | None:
    """Read DateTimeOriginal from EXIF data. Returns date string or None."""
    try:
        exif_data = piexif.load(str(path))
        dt_bytes = exif_data.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
        if dt_bytes:
            return dt_bytes.decode("utf-8")
        # Fallback to DateTime in 0th IFD
        dt_bytes = exif_data.get("0th", {}).get(piexif.ImageIFD.DateTime)
        if dt_bytes:
            return dt_bytes.decode("utf-8")
    except Exception:
        pass
    return None


def parse_exif_date(date_str: str) -> date | None:
    """Parse EXIF date string (YYYY:MM:DD HH:MM:SS) to a date object."""
    try:
        return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S").date()
    except (ValueError, TypeError):
        return None


def write_exif_date(path: Path, dt: date, backup: bool = True) -> None:
    """Write DateTimeOriginal and DateTimeDigitized to the photo's EXIF data."""
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(str(path), str(backup_path))

    date_str = dt.strftime("%Y:%m:%d 12:00:00").encode("utf-8")

    try:
        exif_data = piexif.load(str(path))
    except Exception:
        exif_data = {"0th": {}, "Exif": {}, "1st": {}, "GPS": {}}

    exif_data["Exif"][piexif.ExifIFD.DateTimeOriginal] = date_str
    exif_data["Exif"][piexif.ExifIFD.DateTimeDigitized] = date_str
    exif_data["0th"][piexif.ImageIFD.DateTime] = date_str

    exif_bytes = piexif.dump(exif_data)
    piexif.insert(exif_bytes, str(path))

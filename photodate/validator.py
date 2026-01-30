from datetime import date
from pathlib import Path

from .exif import parse_exif_date, read_exif_date
from .models import PhotoInfo, ValidationIssue

# First practical photograph: ~1826, but let's use 1840 as reasonable cutoff
EARLIEST_PHOTO_DATE = date(1840, 1, 1)

# Maximum gap in days between consecutive photos in an album before warning
MAX_GAP_DAYS = 365


def validate_photo(photo: PhotoInfo) -> list[ValidationIssue]:
    """Check a single photo for date issues."""
    issues: list[ValidationIssue] = []

    if not photo.original_exif_date:
        issues.append(ValidationIssue(
            issue_type="missing_date",
            description="Geen datum gevonden in EXIF data",
            severity="warning",
        ))
        return issues

    dt = parse_exif_date(photo.original_exif_date)
    if dt is None:
        issues.append(ValidationIssue(
            issue_type="invalid_date",
            description=f"Ongeldige datum: {photo.original_exif_date}",
            severity="error",
        ))
        return issues

    today = date.today()

    if dt > today:
        issues.append(ValidationIssue(
            issue_type="future_date",
            description=f"Datum ligt in de toekomst: {dt}",
            severity="error",
        ))

    if dt < EARLIEST_PHOTO_DATE:
        issues.append(ValidationIssue(
            issue_type="too_old",
            description=f"Datum is onrealistisch oud: {dt}",
            severity="error",
        ))

    return issues


def validate_album_sequence(photos: list[PhotoInfo]) -> list[ValidationIssue]:
    """Check for big date gaps between consecutive photos in an album."""
    issues: list[ValidationIssue] = []
    prev_date: date | None = None
    prev_path: Path | None = None

    for photo in photos:
        if not photo.original_exif_date:
            continue
        dt = parse_exif_date(photo.original_exif_date)
        if dt is None:
            continue

        if prev_date is not None:
            gap = abs((dt - prev_date).days)
            if gap > MAX_GAP_DAYS:
                issues.append(ValidationIssue(
                    issue_type="big_gap",
                    description=(
                        f"Grote tijdsprong van {gap} dagen tussen "
                        f"{prev_path.name} ({prev_date}) en "
                        f"{photo.path.name} ({dt})"
                    ),
                    severity="warning",
                ))

        prev_date = dt
        prev_path = photo.path

    return issues


def scan_and_validate(folder: Path) -> list[PhotoInfo]:
    """Scan a folder, read EXIF dates, and validate all photos."""
    extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )

    photos: list[PhotoInfo] = []
    for i, f in enumerate(files):
        photo = PhotoInfo(
            path=f,
            index=i,
            original_exif_date=read_exif_date(f),
        )
        photo.issues = validate_photo(photo)
        photos.append(photo)

    # Album-level validation
    album_issues = validate_album_sequence(photos)
    if album_issues:
        # Attach album-level issues to the first photo as summary
        photos[0].issues.extend(album_issues)

    return photos

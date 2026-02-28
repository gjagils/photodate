"""Perceptual hashing and duplicate detection for photos."""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
from PIL import Image

from .exif import read_exif_date

logger = logging.getLogger(__name__)


@dataclass
class DuplicatePhoto:
    path: Path
    filename: str
    size_bytes: int
    width: int
    height: int
    exif_date: str | None
    phash: str


@dataclass
class DuplicateGroup:
    id: int
    photos: list[DuplicatePhoto] = field(default_factory=list)


def compute_phash(path: Path) -> str:
    """Compute perceptual hash for an image."""
    img = Image.open(path)
    return str(imagehash.phash(img, hash_size=16))


def find_duplicates(
    photo_paths: list[Path],
    cached_hashes: dict[str, str] | None = None,
    threshold: int = 8,
    progress_callback=None,
) -> tuple[list[DuplicateGroup], dict[str, str]]:
    """Find duplicate photos using perceptual hashing.

    Returns (duplicate_groups, updated_hash_cache).
    progress_callback(current, total) is called periodically if provided.
    """
    if cached_hashes is None:
        cached_hashes = {}

    # Compute hashes (use cache where available)
    photos: list[DuplicatePhoto] = []
    updated_cache: dict[str, str] = {}
    total = len(photo_paths)

    for idx, path in enumerate(photo_paths):
        if progress_callback and idx % 50 == 0:
            progress_callback(idx, total)
        fname = path.name
        try:
            # Use cached hash or compute new one
            if fname in cached_hashes:
                hash_str = cached_hashes[fname]
            else:
                hash_str = compute_phash(path)
            updated_cache[fname] = hash_str

            img = Image.open(path)
            w, h = img.size

            photos.append(DuplicatePhoto(
                path=path,
                filename=fname,
                size_bytes=path.stat().st_size,
                width=w,
                height=h,
                exif_date=read_exif_date(path),
                phash=hash_str,
            ))
        except Exception as e:
            logger.warning(f"Could not process {path}: {e}")

    # Compare all pairs
    n = len(photos)
    used = set()
    groups: list[DuplicateGroup] = []
    group_id = 0

    for i in range(n):
        if i in used:
            continue
        hash_i = imagehash.hex_to_hash(photos[i].phash)
        members = [photos[i]]

        for j in range(i + 1, n):
            if j in used:
                continue
            hash_j = imagehash.hex_to_hash(photos[j].phash)
            if hash_i - hash_j <= threshold:
                members.append(photos[j])
                used.add(j)

        if len(members) > 1:
            group_id += 1
            groups.append(DuplicateGroup(id=group_id, photos=members))
            used.add(i)

    return groups, updated_cache

"""JSON-based storage for global settings and per-album data."""
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

STORAGE_DIR = Path("/data")  # Mounted volume in Docker


@dataclass
class FamilyMember:
    name: str
    birthdate: str  # YYYY-MM-DD
    notes: str = ""  # e.g. "draagt bril", "rood haar"


@dataclass
class Milestone:
    photo_index: int  # Which photo number
    date: str  # YYYY-MM-DD
    description: str  # e.g. "Verjaardag papa"


@dataclass
class GlobalSettings:
    family_members: list[FamilyMember] = field(default_factory=list)
    google_credentials_path: str = ""  # Path to Google OAuth2 credentials JSON
    icloud_photos_path: str = ""  # Path to iCloud photos directory

    def save(self) -> None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        path = STORAGE_DIR / "settings.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls) -> "GlobalSettings":
        path = STORAGE_DIR / "settings.json"
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            family_members=[FamilyMember(**m) for m in data.get("family_members", [])],
            google_credentials_path=data.get("google_credentials_path", ""),
            icloud_photos_path=data.get("icloud_photos_path", ""),
        )


@dataclass
class ICloudData:
    """Track dismissed (not-important) iCloud photos per YYYY/MM folder."""
    dismissed: dict[str, list[str]] = field(default_factory=dict)  # "YYYY/MM" -> [filenames]

    def save(self) -> None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        path = STORAGE_DIR / "icloud_data.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls) -> "ICloudData":
        path = STORAGE_DIR / "icloud_data.json"
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            dismissed=data.get("dismissed", {}),
        )


@dataclass
class AlbumData:
    album_rel: str  # Relative path of the album
    start_date: str = ""  # YYYY-MM-DD
    end_date: str = ""  # YYYY-MM-DD
    context_notes: str = ""  # Free text context
    milestones: list[Milestone] = field(default_factory=list)
    # Face clusters: cluster_id -> label (name)
    face_labels: dict[str, str] = field(default_factory=dict)
    # Analysis results stored for the date review page
    photo_dates: dict[str, str] = field(default_factory=dict)  # filename -> YYYY-MM-DD
    photo_reasoning: dict[str, str] = field(default_factory=dict)  # filename -> reason
    photo_confidence: dict[str, str] = field(default_factory=dict)  # filename -> level
    photo_groups: dict[str, int] = field(default_factory=dict)  # filename -> group_id
    # Perceptual hashes for duplicate detection
    photo_hashes: dict[str, str] = field(default_factory=dict)  # filename -> phash hex
    # Whether all photos have EXIF dates (user-managed flag)
    exif_complete: bool = False
    # Whether this album should never be organized (user-managed flag)
    never_organize: bool = False

    def save(self) -> None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        # Use album_rel as filename, replace / with __
        safe_name = self.album_rel.replace("/", "__").replace("\\", "__")
        path = STORAGE_DIR / f"album_{safe_name}.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, album_rel: str) -> "AlbumData":
        safe_name = album_rel.replace("/", "__").replace("\\", "__")
        path = STORAGE_DIR / f"album_{safe_name}.json"
        if not path.exists():
            return cls(album_rel=album_rel)
        data = json.loads(path.read_text())
        return cls(
            album_rel=data.get("album_rel", album_rel),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            context_notes=data.get("context_notes", ""),
            milestones=[Milestone(**m) for m in data.get("milestones", [])],
            face_labels=data.get("face_labels", {}),
            photo_dates=data.get("photo_dates", {}),
            photo_reasoning=data.get("photo_reasoning", {}),
            photo_confidence=data.get("photo_confidence", {}),
            photo_groups=data.get("photo_groups", {}),
            photo_hashes=data.get("photo_hashes", {}),
            exif_complete=data.get("exif_complete", False),
            never_organize=data.get("never_organize", False),
        )

"""JSON-based storage for global settings and per-album data."""
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

STORAGE_DIR = Path(os.environ.get("DATA_DIR", "/data"))

_lock = threading.Lock()


@dataclass
class FamilyMember:
    name: str
    birthdate: str  # YYYY-MM-DD
    notes: str = ""


@dataclass
class Milestone:
    photo_index: int
    date: str  # YYYY-MM-DD
    description: str


@dataclass
class FaceProfile:
    """Named face encoding saved globally for reuse across albums."""
    name: str
    encoding: list[float]  # 128-dim face_recognition vector


@dataclass
class GlobalSettings:
    family_members: list[FamilyMember] = field(default_factory=list)
    face_profiles: list[FaceProfile] = field(default_factory=list)

    def save(self) -> None:
        with _lock:
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            path = STORAGE_DIR / "settings.json"
            path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls) -> "GlobalSettings":
        path = STORAGE_DIR / "settings.json"
        if not path.exists():
            return cls()
        with _lock:
            data = json.loads(path.read_text())
        return cls(
            family_members=[FamilyMember(**m) for m in data.get("family_members", [])],
            face_profiles=[FaceProfile(**p) for p in data.get("face_profiles", [])],
        )


@dataclass
class AlbumData:
    album_rel: str
    start_date: str = ""
    end_date: str = ""
    context_notes: str = ""
    milestones: list[Milestone] = field(default_factory=list)
    face_labels: dict[str, str] = field(default_factory=dict)
    photo_dates: dict[str, str] = field(default_factory=dict)       # filename -> YYYY-MM-DD
    photo_reasoning: dict[str, str] = field(default_factory=dict)   # filename -> reason
    photo_confidence: dict[str, str] = field(default_factory=dict)  # filename -> level
    photo_groups: dict[str, int] = field(default_factory=dict)      # filename -> group_id
    photo_group_labels: dict[str, str] = field(default_factory=dict) # str(group_id) -> label
    photo_date_source: dict[str, str] = field(default_factory=dict)  # filename -> "spreiding" | "ai" | "handmatig"
    not_mine_photos: list[str] = field(default_factory=list)
    # Persisted face detection results (so results page works after page reload)
    face_clusters: list[dict] = field(default_factory=list)
    faces_per_photo: dict[str, list[str]] = field(default_factory=dict)  # str(photo_idx) -> [cluster_ids]
    # Analysis lifecycle
    analysis_status: str = "idle"   # "idle" | "running" | "done" | "error"
    analysis_error: str = ""
    applied_count: int = 0           # how many photos had dates written

    def save(self) -> None:
        with _lock:
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = self.album_rel.replace("/", "__").replace("\\", "__")
            path = STORAGE_DIR / f"album_{safe_name}.json"
            path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, album_rel: str) -> "AlbumData":
        safe_name = album_rel.replace("/", "__").replace("\\", "__")
        path = STORAGE_DIR / f"album_{safe_name}.json"
        if not path.exists():
            return cls(album_rel=album_rel)
        with _lock:
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
            photo_group_labels=data.get("photo_group_labels", {}),
            photo_date_source=data.get("photo_date_source", {}),
            not_mine_photos=data.get("not_mine_photos", []),
            face_clusters=data.get("face_clusters", []),
            faces_per_photo=data.get("faces_per_photo", {}),
            analysis_status=data.get("analysis_status", "idle"),
            analysis_error=data.get("analysis_error", ""),
            applied_count=data.get("applied_count", 0),
        )

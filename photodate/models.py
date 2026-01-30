from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class DateEstimate:
    date: date
    confidence: str  # "high", "medium", "low"
    reasoning: str


@dataclass
class ValidationIssue:
    issue_type: str  # "future_date", "too_old", "big_gap", "context_mismatch"
    description: str
    severity: str  # "error", "warning"


@dataclass
class PhotoInfo:
    path: Path
    index: int
    original_exif_date: str | None = None
    estimate: DateEstimate | None = None
    issues: list[ValidationIssue] = field(default_factory=list)
    approved: bool = False


@dataclass
class Album:
    folder: Path
    context: str
    photos: list[PhotoInfo] = field(default_factory=list)

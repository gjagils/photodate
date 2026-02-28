"""Google Photos API integration for photo verification."""
import json
import logging
from datetime import datetime
from pathlib import Path

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from .storage import STORAGE_DIR

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/photoslibrary.readonly"]
TOKEN_PATH = STORAGE_DIR / "google_token.json"


def get_oauth_flow(credentials_path: str, redirect_uri: str) -> Flow:
    """Create an OAuth2 flow for Google Photos authorization."""
    flow = Flow.from_client_secrets_file(
        credentials_path,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )
    return flow


def save_credentials(creds: Credentials) -> None:
    """Save OAuth2 credentials to disk."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())


def load_credentials() -> Credentials | None:
    """Load saved OAuth2 credentials, refreshing if needed."""
    if not TOKEN_PATH.exists():
        return None
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(GoogleRequest())
        save_credentials(creds)
    if creds and creds.valid:
        return creds
    return None


def get_service(creds: Credentials):
    """Build a Google Photos API service."""
    return build("photoslibrary", "v1", credentials=creds, static_discovery=False)


def list_all_media_items(service) -> list[dict]:
    """Fetch all media items from Google Photos.

    Returns list of dicts with: id, filename, creation_time, width, height.
    """
    items = []
    page_token = None

    while True:
        body = {"pageSize": 100}
        if page_token:
            body["pageToken"] = page_token

        response = service.mediaItems().list(**body).execute()
        for item in response.get("mediaItems", []):
            metadata = item.get("mediaMetadata", {})
            creation_time = metadata.get("creationTime", "")
            items.append({
                "id": item["id"],
                "filename": item.get("filename", ""),
                "creation_time": creation_time,
                "width": int(metadata.get("width", 0)),
                "height": int(metadata.get("height", 0)),
            })

        page_token = response.get("nextPageToken")
        if not page_token:
            break

        logger.info(f"Fetched {len(items)} Google Photos items so far...")

    logger.info(f"Total Google Photos items: {len(items)}")
    return items


def _parse_datetime(dt_str: str) -> datetime | None:
    """Parse various datetime formats."""
    for fmt in [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y:%m:%d %H:%M:%S",
    ]:
        try:
            return datetime.strptime(dt_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def match_local_to_google(
    local_photos: list[dict],
    google_items: list[dict],
) -> dict:
    """Match local NAS photos against Google Photos items.

    local_photos: list of {filename, exif_date, width, height, year}
    google_items: list from list_all_media_items()

    Returns {
        "matched": [{"local": ..., "google": ...}],
        "unmatched": [{"local": ...}],
        "by_year": {year: {"total": N, "matched": M, "unmatched_files": [...]}},
    }
    """
    # Build lookup index from Google Photos by date+dimensions
    google_by_date = {}
    for item in google_items:
        dt = _parse_datetime(item["creation_time"])
        if dt:
            key = (dt.strftime("%Y-%m-%d"), item["width"], item["height"])
            google_by_date.setdefault(key, []).append(item)
            # Also index by just date (less strict match)
            date_key = dt.strftime("%Y-%m-%d")
            google_by_date.setdefault(("date_only", date_key), []).append(item)

    matched = []
    unmatched = []

    for local in local_photos:
        local_dt = _parse_datetime(local.get("exif_date", "") or "")
        found = False

        if local_dt:
            # Try exact match: date + dimensions
            key = (local_dt.strftime("%Y-%m-%d"), local["width"], local["height"])
            if key in google_by_date:
                matched.append({"local": local, "google": google_by_date[key][0]})
                found = True
            else:
                # Try date-only match (dimensions might differ due to re-encoding)
                date_key = ("date_only", local_dt.strftime("%Y-%m-%d"))
                candidates = google_by_date.get(date_key, [])
                for candidate in candidates:
                    # Accept if filename matches or dimensions are close
                    if (candidate["filename"] == local["filename"] or
                            abs(candidate["width"] - local["width"]) <= 2 and
                            abs(candidate["height"] - local["height"]) <= 2):
                        matched.append({"local": local, "google": candidate})
                        found = True
                        break

        if not found:
            unmatched.append({"local": local})

    # Aggregate by year
    by_year = {}
    for m in matched:
        year = m["local"].get("year", "onbekend")
        by_year.setdefault(year, {"total": 0, "matched": 0, "unmatched_files": []})
        by_year[year]["total"] += 1
        by_year[year]["matched"] += 1

    for u in unmatched:
        year = u["local"].get("year", "onbekend")
        by_year.setdefault(year, {"total": 0, "matched": 0, "unmatched_files": []})
        by_year[year]["total"] += 1
        by_year[year]["unmatched_files"].append(u["local"]["filename"])

    return {
        "matched": matched,
        "unmatched": unmatched,
        "by_year": dict(sorted(by_year.items())),
        "total": len(matched) + len(unmatched),
        "total_matched": len(matched),
    }

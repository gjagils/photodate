"""Smoke tests: verify every main page returns 200, not 500."""
import pytest
from starlette.testclient import TestClient

from photodate.web import app

client = TestClient(app, raise_server_exceptions=False)

PAGES = [
    "/",
    "/icloud",
    "/settings",
    "/organize",
    "/gphotos-local",
]


@pytest.mark.parametrize("path", PAGES)
def test_page_returns_200(path):
    resp = client.get(path)
    assert resp.status_code == 200, (
        f"GET {path} → HTTP {resp.status_code}\n"
        f"{resp.text[:400]}"
    )


def test_icloud_counts_api_without_path():
    """API returns 400 when no iCloud path is configured (not 500)."""
    resp = client.get("/api/icloud/2023/counts")
    assert resp.status_code in (200, 400, 404)


def test_gphotos_counts_api_without_path():
    """API returns 400 when no Google download path is configured (not 500)."""
    resp = client.get("/api/gphotos-local/counts")
    assert resp.status_code in (200, 400, 404)

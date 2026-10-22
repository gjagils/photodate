"""Tests for _build_context in analyzer.py."""
import pytest

from photodate.analyzer import _build_context
from photodate.storage import AlbumData, FamilyMember, GlobalSettings, Milestone


def test_empty_context():
    album = AlbumData(album_rel="test")
    settings = GlobalSettings()
    result = _build_context(album, settings)
    assert result == ""


def test_context_notes():
    album = AlbumData(album_rel="test", context_notes="Bruiloft van oom Piet")
    settings = GlobalSettings()
    result = _build_context(album, settings)
    assert "Bruiloft van oom Piet" in result


def test_date_range_both():
    album = AlbumData(album_rel="test", start_date="1985-07-01", end_date="1985-08-31")
    settings = GlobalSettings()
    result = _build_context(album, settings)
    assert "1985-07-01" in result
    assert "1985-08-31" in result


def test_date_range_start_only():
    album = AlbumData(album_rel="test", start_date="1985-07-01")
    settings = GlobalSettings()
    result = _build_context(album, settings)
    assert "1985-07-01" in result
    assert "eindigt" not in result


def test_date_range_end_only():
    album = AlbumData(album_rel="test", end_date="1985-08-31")
    settings = GlobalSettings()
    result = _build_context(album, settings)
    assert "1985-08-31" in result


def test_milestones_included():
    album = AlbumData(
        album_rel="test",
        milestones=[Milestone(photo_index=5, date="1985-12-25", description="Kerstavond")],
    )
    settings = GlobalSettings()
    result = _build_context(album, settings)
    assert "Foto 5" in result
    assert "Kerstavond" in result
    assert "1985-12-25" in result


def test_face_labels_included():
    album = AlbumData(album_rel="test", face_labels={"0": "Jan", "1": "Marie"})
    settings = GlobalSettings()
    result = _build_context(album, settings)
    assert "Jan" in result
    assert "Marie" in result


def test_family_members_included():
    settings = GlobalSettings(
        family_members=[FamilyMember(name="Piet", birthdate="1940-06-10", notes="baard")]
    )
    album = AlbumData(album_rel="test")
    result = _build_context(album, settings)
    assert "Piet" in result
    assert "1940-06-10" in result
    assert "baard" in result


def test_family_member_without_notes():
    settings = GlobalSettings(
        family_members=[FamilyMember(name="Mia", birthdate="1945-01-01")]
    )
    album = AlbumData(album_rel="test")
    result = _build_context(album, settings)
    assert "Mia" in result
    # no em dash for empty notes
    assert "—" not in result

"""Tests for storage.py: AlbumData and GlobalSettings load/save."""
import pytest

from photodate.storage import AlbumData, FaceProfile, FamilyMember, GlobalSettings, Milestone


class TestGlobalSettings:
    def test_save_and_load_empty(self, tmp_data_dir):
        s = GlobalSettings()
        s.save()
        loaded = GlobalSettings.load()
        assert loaded.family_members == []
        assert loaded.face_profiles == []

    def test_save_and_load_family_members(self, tmp_data_dir):
        s = GlobalSettings(
            family_members=[FamilyMember(name="Jan", birthdate="1950-03-15", notes="bril")]
        )
        s.save()
        loaded = GlobalSettings.load()
        assert len(loaded.family_members) == 1
        m = loaded.family_members[0]
        assert m.name == "Jan"
        assert m.birthdate == "1950-03-15"
        assert m.notes == "bril"

    def test_save_and_load_face_profiles(self, tmp_data_dir):
        enc = [0.1] * 128
        s = GlobalSettings(face_profiles=[FaceProfile(name="Marie", encoding=enc)])
        s.save()
        loaded = GlobalSettings.load()
        assert len(loaded.face_profiles) == 1
        assert loaded.face_profiles[0].name == "Marie"
        assert len(loaded.face_profiles[0].encoding) == 128

    def test_load_missing_file_returns_defaults(self, tmp_data_dir):
        loaded = GlobalSettings.load()
        assert isinstance(loaded, GlobalSettings)
        assert loaded.family_members == []

    def test_data_dir_env_var(self, tmp_path, monkeypatch):
        import photodate.storage as storage
        monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "custom")
        s = GlobalSettings(family_members=[FamilyMember(name="X", birthdate="2000-01-01")])
        s.save()
        assert (tmp_path / "custom" / "settings.json").exists()


class TestAlbumData:
    def test_save_and_load_roundtrip(self, tmp_data_dir):
        album = AlbumData(
            album_rel="Vakantie/1985",
            start_date="1985-07-01",
            end_date="1985-08-31",
            photo_dates={"foto1.jpg": "1985-07-15"},
            analysis_status="done",
            applied_count=3,
        )
        album.save()
        loaded = AlbumData.load("Vakantie/1985")
        assert loaded.start_date == "1985-07-01"
        assert loaded.photo_dates == {"foto1.jpg": "1985-07-15"}
        assert loaded.analysis_status == "done"
        assert loaded.applied_count == 3

    def test_load_missing_album_returns_defaults(self, tmp_data_dir):
        loaded = AlbumData.load("does/not/exist")
        assert loaded.album_rel == "does/not/exist"
        assert loaded.analysis_status == "idle"
        assert loaded.photo_dates == {}

    def test_slash_in_album_rel_is_safe(self, tmp_data_dir):
        album = AlbumData(album_rel="a/b/c", start_date="2000-01-01")
        album.save()
        loaded = AlbumData.load("a/b/c")
        assert loaded.start_date == "2000-01-01"

    def test_milestones_roundtrip(self, tmp_data_dir):
        album = AlbumData(
            album_rel="test",
            milestones=[Milestone(photo_index=3, date="1990-12-25", description="Kerst")],
        )
        album.save()
        loaded = AlbumData.load("test")
        assert len(loaded.milestones) == 1
        ms = loaded.milestones[0]
        assert ms.photo_index == 3
        assert ms.date == "1990-12-25"
        assert ms.description == "Kerst"

    def test_face_clusters_roundtrip(self, tmp_data_dir):
        clusters = [{"cluster_id": 0, "photos": [0, 1], "centroid": [0.5] * 128}]
        album = AlbumData(album_rel="test", face_clusters=clusters)
        album.save()
        loaded = AlbumData.load("test")
        assert loaded.face_clusters[0]["cluster_id"] == 0

    def test_photo_group_labels_roundtrip(self, tmp_data_dir):
        album = AlbumData(album_rel="test", photo_group_labels={"0": "Zomer"})
        album.save()
        loaded = AlbumData.load("test")
        assert loaded.photo_group_labels == {"0": "Zomer"}

    def test_not_mine_photos_roundtrip(self, tmp_data_dir):
        album = AlbumData(album_rel="test", not_mine_photos=["foto_vreemd.jpg"])
        album.save()
        loaded = AlbumData.load("test")
        assert "foto_vreemd.jpg" in loaded.not_mine_photos

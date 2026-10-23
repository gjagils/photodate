import os
import tempfile
from pathlib import Path

import pytest

# Point storage at a temp dir so tests don't touch /data
@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Re-evaluate the module-level STORAGE_DIR so saves land in tmp_path
    import photodate.storage as storage
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def sample_jpg(tmp_path) -> Path:
    """A minimal valid JPEG for EXIF tests."""
    from PIL import Image
    path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 80), color=(128, 64, 32))
    img.save(path, format="JPEG")
    return path

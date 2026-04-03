"""Test configuration: patch filesystem paths before any app imports."""
import os
from pathlib import Path

# Temp dirs that survive the test session
_data_dir = Path("/tmp/photodate_test_data")
_photos_dir = Path("/tmp/photodate_test_photos")
_data_dir.mkdir(exist_ok=True)
_photos_dir.mkdir(exist_ok=True)

# Must be set before importing photodate modules
os.environ["PHOTOS_PATHS"] = str(_photos_dir)
os.environ["OPENAI_API_KEY"] = "sk-test-not-used"

# Patch STORAGE_DIR before web.py imports it from storage
import photodate.storage
photodate.storage.STORAGE_DIR = _data_dir

import photodate.web
photodate.web.STORAGE_DIR = _data_dir

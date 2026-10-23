"""Tests for events.py with a fake vision client."""
import json
from types import SimpleNamespace

from PIL import Image

from photodate import events


class FakeClient:
    def __init__(self, relation="moment", raw=None):
        self.calls = []
        self.relation = relation
        self.raw = raw
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, **kwargs):
        images = [c for c in messages[0]["content"] if c["type"] == "image_url"]
        self.calls.append(len(images))
        n = len(images)
        body = self.raw if self.raw is not None else json.dumps({
            "overgangen": [self.relation] * (n - 1),
            "labels": ["Feest"] * n,
            "seizoenen": ["winter"] * n,
            "feestdagen": ["verjaardag"] * n,
            "leeftijden": [9] * n,
        })
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))])


def _photos(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"boek_{i:03d}.jpg"
        Image.new("RGB", (60, 40), (i * 9 % 255, 80, 120)).save(p)
        paths.append(p)
    return paths


def test_batches_overlap_so_every_transition_is_judged(tmp_path):
    client = FakeClient("moment")
    result = events.detect_links(_photos(tmp_path, 25), client, "IX", "okt 1988 – jan 1990")
    assert client.calls == [12, 12, 3]
    assert result.links == ["nieuw"] + ["moment"] * 24
    assert all(result.labels)
    assert result.seasons == ["winter"] * 25
    assert result.holidays == ["verjaardag"] * 25
    assert result.ages == [9] * 25


def test_unknown_values_fall_back(tmp_path):
    raw = json.dumps({
        "overgangen": ["moment", "zelfde dag?"], "labels": [], "seizoenen": ["hittegolf"],
        "feestdagen": ["carnaval", "sinterklaas"], "leeftijden": ["negen", 250, "50"],
    })
    result = events.detect_links(_photos(tmp_path, 5), FakeClient(raw=raw), "IX", "?")
    assert result.links == ["nieuw", "moment", "nieuw", "nieuw", "nieuw"]
    assert result.labels == [""] * 5
    assert result.seasons == ["onbekend"] * 5
    assert result.holidays == ["geen", "sinterklaas", "geen", "geen", "geen"]
    assert result.ages == [None, None, 50, None, None]


def test_unreadable_answer_does_not_crash(tmp_path):
    result = events.detect_links(_photos(tmp_path, 3), FakeClient(raw="geen json"), "IX", "?")
    assert result.links == ["nieuw"] * 3


def test_single_photo_needs_no_call(tmp_path):
    client = FakeClient()
    result = events.detect_links(_photos(tmp_path, 1), client, "IX", "?")
    assert result == (["nieuw"], [""], ["onbekend"], ["geen"], [None])
    assert client.calls == []

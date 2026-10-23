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
        body = self.raw if self.raw is not None else json.dumps({
            "overgangen": [self.relation] * (len(images) - 1),
            "labels": ["Camping"] * len(images),
            "seizoenen": ["zomer"] * len(images),
            "rotaties": [180] * len(images),
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
    links, labels, seasons, rotations = events.detect_links(_photos(tmp_path, 25), client, "IX", "okt 1988 – jan 1990")
    assert client.calls == [12, 12, 3]
    assert links[0] == "nieuw"
    assert links[1:] == ["moment"] * 24
    assert all(labels)
    assert seasons == ["zomer"] * 25
    assert rotations == [180] * 25


def test_unknown_relations_and_short_answers_fall_back_to_new(tmp_path):
    client = FakeClient(raw=json.dumps({"overgangen": ["moment", "zelfde dag?"], "labels": [], "seizoenen": ["hittegolf"], "rotaties": [45, "90", None]}))
    links, labels, seasons, rotations = events.detect_links(_photos(tmp_path, 5), client, "IX", "?")
    assert links == ["nieuw", "moment", "nieuw", "nieuw", "nieuw"]
    assert labels == [""] * 5
    assert seasons == ["onbekend"] * 5
    assert rotations == [0, 90, 0, 0, 0]


def test_unreadable_answer_does_not_crash(tmp_path):
    links = events.detect_links(_photos(tmp_path, 3), FakeClient(raw="geen json"), "IX", "?").links
    assert links == ["nieuw"] * 3


def test_single_photo_needs_no_call(tmp_path):
    client = FakeClient()
    assert events.detect_links(_photos(tmp_path, 1), client, "IX", "?") == (["nieuw"], [""], ["onbekend"], [0])
    assert client.calls == []

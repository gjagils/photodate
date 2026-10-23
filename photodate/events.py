"""Judge how consecutive album photos belong together (same moment, same trip, new event)."""
import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

RELATIONS = ("moment", "reeks", "nieuw")
EVENTS_MODEL = os.environ.get("OPENAI_EVENTS_MODEL", "gpt-4o")
BATCH_SIZE = 12
THUMB_SIZE = 512

PROMPT = """\
Dit zijn {n} ingescande foto's uit een fysiek familiefotoalbum, in de volgorde waarin ze \
in het album zitten. Album: "{album}", periode {period}.

Beoordeel voor elke overgang van een foto naar de volgende hoe ze samenhangen:
- "moment": zelfde dag en gelegenheid (zelfde plek, mensen, kleding, licht; bijv. een feest, \
uitje of één middag op de camping)
- "reeks": zelfde reeks maar waarschijnlijk een andere dag (bijv. dezelfde vakantie of logeerpartij)
- "nieuw": een andere gebeurtenis

Geef ook per foto een kort label van de gebeurtenis (hooguit 4 woorden, Nederlands).

Antwoord alleen met JSON in deze vorm:
{{"overgangen": [{m} waarden uit "moment", "reeks", "nieuw"], "labels": [{n} korte labels]}}
"""


def _encode(path: Path) -> str:
    img = ImageOps.exif_transpose(Image.open(path))
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode()


def classify_batch(client, paths: list[Path], album: str, period: str, model: str = EVENTS_MODEL
                   ) -> tuple[list[str], list[str]]:
    """Return (relations between consecutive photos, one label per photo) for one batch."""
    n = len(paths)
    content: list[dict] = [{"type": "text", "text": PROMPT.format(n=n, m=n - 1, album=album, period=period)}]
    for i, path in enumerate(paths, 1):
        content.append({"type": "text", "text": f"Foto {i}"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_encode(path)}", "detail": "low"},
        })
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        data = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        logger.warning("Onleesbaar antwoord voor gebeurtenissen in %s", album)
        data = {}
    relations = [r if r in RELATIONS else "nieuw" for r in data.get("overgangen", [])][: n - 1]
    relations += ["nieuw"] * (n - 1 - len(relations))
    labels = [str(l)[:60] for l in data.get("labels", [])][:n]
    labels += [""] * (n - len(labels))
    return relations, labels


def detect_links(
    paths: list[Path],
    client,
    album: str,
    period: str,
    model: str = EVENTS_MODEL,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[list[str], list[str]]:
    """links[i] is how photo i relates to photo i-1 (links[0] is always "nieuw")."""
    n = len(paths)
    links = ["nieuw"] * n
    labels = [""] * n
    start = 0
    while start < n - 1:
        end = min(start + BATCH_SIZE, n)
        relations, batch_labels = classify_batch(client, paths[start:end], album, period, model)
        for k, relation in enumerate(relations):
            links[start + k + 1] = relation
        for k, label in enumerate(batch_labels):
            if not labels[start + k]:
                labels[start + k] = label
        if progress:
            progress(end, n)
        # Batches overlap by one photo so every transition is judged with context on both sides.
        start = end - 1
    return links, labels

"""Judge how consecutive album photos belong together, their season and how they are turned."""
import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Callable, NamedTuple, Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

RELATIONS = ("moment", "reeks", "nieuw")
SEASONS = ("lente", "zomer", "herfst", "winter", "onbekend")
ROTATIONS = (0, 90, 180, 270)
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

Schat per foto het seizoen (Nederland), alleen op duidelijke aanwijzingen:
- "zomer": zwemkleding, korte broeken en blote armen in de zon, strand, volle zomerse begroeiing
- "herfst": verkleurd of vallend blad, kale bomen met jassen maar zonder sneeuw, Sint-Maarten
- "winter": sneeuw, ijs, dikke winterjassen en mutsen, kerstboom of kerstversiering
- "lente": bloesem, bollenvelden, jong blad, Pasen
- "onbekend": binnenopnames of te weinig aanwijzingen. Gebruik dit liever dan te gokken.

Geef per foto hoeveel graden hij met de klok mee gedraaid moet worden om rechtop te staan: \
0, 90, 180 of 270. Let op mensen en gezichten, de horizon, gebouwen en tekst. \
Staan mensen op hun kop, dan is het 180. Twijfel je, geef 0.

Antwoord alleen met JSON in deze vorm:
{{"overgangen": [{m} waarden uit "moment", "reeks", "nieuw"], "labels": [{n} korte labels], \
"seizoenen": [{n} waarden uit "lente", "zomer", "herfst", "winter", "onbekend"], \
"rotaties": [{n} waarden uit 0, 90, 180, 270]}}
"""


class AlbumAnalysis(NamedTuple):
    links: list[str]      # relation of photo i to photo i-1 (links[0] is "nieuw")
    labels: list[str]
    seasons: list[str]
    rotations: list[int]  # clockwise degrees needed on the image as displayed


def _encode(path: Path) -> str:
    img = ImageOps.exif_transpose(Image.open(path))
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _rotation(value) -> int:
    try:
        r = int(value)
    except (TypeError, ValueError):
        return 0
    return r if r in ROTATIONS else 0


def classify_batch(client, paths: list[Path], album: str, period: str, model: str = EVENTS_MODEL
                   ) -> AlbumAnalysis:
    """Analyse one batch; links[0] of the result is a placeholder for the batch's first photo."""
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

    def fit(values: list, length: int, fill):
        return values[:length] + [fill] * (length - len(values[:length]))

    relations = fit([r if r in RELATIONS else "nieuw" for r in data.get("overgangen", [])], n - 1, "nieuw")
    labels = fit([str(l)[:60] for l in data.get("labels", [])], n, "")
    seasons = fit([s if s in SEASONS else "onbekend" for s in data.get("seizoenen", [])], n, "onbekend")
    rotations = fit([_rotation(r) for r in data.get("rotaties", [])], n, 0)
    return AlbumAnalysis(["nieuw"] + relations, labels, seasons, rotations)


def detect_links(
    paths: list[Path],
    client,
    album: str,
    period: str,
    model: str = EVENTS_MODEL,
    progress: Optional[Callable[[int, int], None]] = None,
) -> AlbumAnalysis:
    n = len(paths)
    result = AlbumAnalysis(["nieuw"] * n, [""] * n, ["onbekend"] * n, [0] * n)
    seen = [False] * n
    start = 0
    while start < n - 1:
        end = min(start + BATCH_SIZE, n)
        batch = classify_batch(client, paths[start:end], album, period, model)
        for k in range(end - start):
            i = start + k
            if k > 0:
                result.links[i] = batch.links[k]
            if not seen[i]:
                result.labels[i] = batch.labels[k]
                result.seasons[i] = batch.seasons[k]
                result.rotations[i] = batch.rotations[k]
                seen[i] = True
        if progress:
            progress(end, n)
        # Batches overlap by one photo so every transition is judged with context on both sides.
        start = end - 1
    return result

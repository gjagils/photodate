"""Judge how consecutive album photos belong together, their season and any holiday."""
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
HOLIDAYS = ("sinterklaas", "kerst", "oudjaar", "pasen", "koninginnedag", "verjaardag", "geen")
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

Geef per foto of het om een Nederlandse feestdag gaat, alleen bij duidelijke aanwijzingen:
- "sinterklaas": Sinterklaas of Pieten, surprises, gedichten, schoen zetten, pakjesavond, intocht
- "kerst": kerstboom met cadeaus, kerstdiner, kerststal
- "oudjaar": vuurwerk, oliebollen, proosten om middernacht
- "pasen": paaseieren, paashaas, eieren zoeken
- "koninginnedag": oranje kleding of versiering, vrijmarkt, vlaggen, kinderspelen op straat
- "verjaardag": taart met kaarsjes, cadeaus uitpakken, slingers, ballonnen, kring met visite
- "geen": anders

Geef bij een verjaardag de leeftijd van de jarige als die echt zichtbaar is (aantal kaarsjes, \
een cijfer op taart, ballon, slinger of kaart; Abraham of Sarah is 50). Anders null.

Antwoord alleen met JSON in deze vorm:
{{"overgangen": [{m} waarden uit "moment", "reeks", "nieuw"], "labels": [{n} korte labels], \
"seizoenen": [{n} waarden uit "lente", "zomer", "herfst", "winter", "onbekend"], \
"feestdagen": [{n} waarden uit "sinterklaas", "kerst", "oudjaar", "pasen", "koninginnedag", \
"verjaardag", "geen"], "leeftijden": [{n} getallen of null]}}
"""


class AlbumAnalysis(NamedTuple):
    links: list[str]               # relation of photo i to photo i-1 (links[0] is "nieuw")
    labels: list[str]
    seasons: list[str]
    holidays: list[str]
    ages: list[Optional[int]]      # age of the birthday person, when visible


def _encode(path: Path) -> str:
    img = ImageOps.exif_transpose(Image.open(path))
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _age(value) -> Optional[int]:
    try:
        age = int(value)
    except (TypeError, ValueError):
        return None
    return age if 0 < age < 110 else None


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
        values = values[:length]
        return values + [fill] * (length - len(values))

    relations = fit([r if r in RELATIONS else "nieuw" for r in data.get("overgangen", [])], n - 1, "nieuw")
    labels = fit([str(l)[:60] for l in data.get("labels", [])], n, "")
    seasons = fit([s if s in SEASONS else "onbekend" for s in data.get("seizoenen", [])], n, "onbekend")
    holidays = fit([h if h in HOLIDAYS else "geen" for h in data.get("feestdagen", [])], n, "geen")
    ages = fit([_age(a) for a in data.get("leeftijden", [])], n, None)
    return AlbumAnalysis(["nieuw"] + relations, labels, seasons, holidays, ages)


def detect_links(
    paths: list[Path],
    client,
    album: str,
    period: str,
    model: str = EVENTS_MODEL,
    progress: Optional[Callable[[int, int], None]] = None,
) -> AlbumAnalysis:
    n = len(paths)
    result = AlbumAnalysis(["nieuw"] * n, [""] * n, ["onbekend"] * n, ["geen"] * n, [None] * n)
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
                result.holidays[i] = batch.holidays[k]
                result.ages[i] = batch.ages[k]
                seen[i] = True
        if progress:
            progress(end, n)
        # Batches overlap by one photo so every transition is judged with context on both sides.
        start = end - 1
    return result

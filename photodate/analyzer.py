import base64
import json
import re
from datetime import date
from io import BytesIO

import openai
from PIL import Image

from .models import DateEstimate, PhotoInfo
from .storage import AlbumData, GlobalSettings

MAX_IMAGE_SIZE = 1024
BATCH_SIZE = 10


def _resize_and_encode(path) -> tuple[str, str]:
    """Resize image to max MAX_IMAGE_SIZE px and return base64 + media type."""
    img = Image.open(path)
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


def _build_context(album_data: AlbumData, settings: GlobalSettings) -> str:
    """Build a rich context string from album data and global settings."""
    parts = []

    if album_data.context_notes:
        parts.append(f"Albumbeschrijving: {album_data.context_notes}")

    if album_data.start_date and album_data.end_date:
        parts.append(f"Periode album: {album_data.start_date} tot {album_data.end_date}")
    elif album_data.start_date:
        parts.append(f"Album begint rond: {album_data.start_date}")
    elif album_data.end_date:
        parts.append(f"Album eindigt rond: {album_data.end_date}")

    if album_data.milestones:
        ms_lines = []
        for m in album_data.milestones:
            ms_lines.append(f"  - Foto {m.photo_index}: {m.description} ({m.date})")
        parts.append("IJkpunten:\n" + "\n".join(ms_lines))

    if album_data.face_labels:
        face_lines = []
        for fid, name in album_data.face_labels.items():
            face_lines.append(f"  - {fid}: {name}")
        parts.append("Geïdentificeerde personen:\n" + "\n".join(face_lines))

    if settings.family_members:
        fam_lines = []
        for m in settings.family_members:
            line = f"  - {m.name} (geboren {m.birthdate})"
            if m.notes:
                line += f" — {m.notes}"
            fam_lines.append(line)
        parts.append("Familieleden:\n" + "\n".join(fam_lines))

    return "\n\n".join(parts)


PROMPT_TEMPLATE = """\
Je analyseert ingescande foto's uit een fysiek fotoalbum van een Nederlandse familie.

{context}

Dit zijn foto's {start} t/m {end} van {total} in het album, in chronologische volgorde
(ze zijn zo ingescand).

Schat voor elke foto de datum waarop de foto oorspronkelijk is gemaakt. Let op:
- Kledingstijlen, kapsels, leeftijd van personen
- Auto's, technologie, apparaten, meubels
- Omgeving, seizoenaanwijzingen (vegetatie, licht, weer)
- Nederlandse feestdagen en gewoontes: Sinterklaas (5 dec), Kerst, Oud & Nieuw,
  Carnaval (feb/mrt), Koninginnedag (30 apr), Pasen, zomervakantie (jul/aug)
- Verjaardagen van genoemde familieleden
- Foto's staan in chronologische volgorde — de datum mag niet eerder zijn dan
  de vorige foto
- Groepeer foto's die duidelijk van dezelfde gelegenheid/dag zijn

Als je alleen een seizoen of jaar kunt schatten, gebruik dan het midden van die
periode (bijv. 15 juli voor "zomer 1987").

Antwoord in JSON (geen markdown codeblock):
[{{"photo_index": N, "estimated_date": "YYYY-MM-DD", \
"confidence": "high|medium|low", \
"reasoning": "...", \
"group": G, \
"group_label": "beschrijving van de gelegenheid"}}]

group: een nummer (start bij 1) dat aangeeft welke foto's bij dezelfde
gelegenheid/dag horen. Foto's van dezelfde dag/event krijgen hetzelfde nummer.
group_label: korte beschrijving zoals "Kerst thuis", "Strandvakantie", "Verjaardag".
"""


def analyze_batch(
    photos: list[PhotoInfo],
    context: str,
    client: openai.OpenAI,
    start_offset: int = 0,
    total: int = 0,
) -> list[dict]:
    """Send a batch of photos to OpenAI for date estimation."""
    content: list[dict] = []

    for photo in photos:
        b64, media_type = _resize_and_encode(photo.path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        })
        content.append({
            "type": "text",
            "text": f"Foto {photo.index + 1}: {photo.path.name}",
        })

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        start=start_offset + 1,
        end=start_offset + len(photos),
        total=total or len(photos),
    )
    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )

    response_text = response.choices[0].message.content
    # Strip markdown code fences if present
    response_text = re.sub(r"^```(?:json)?\n?", "", response_text)
    response_text = re.sub(r"\n?```$", "", response_text)

    return json.loads(response_text)


def analyze_album_full(
    photos: list[PhotoInfo],
    album_data: AlbumData,
    settings: GlobalSettings,
    client: openai.OpenAI,
) -> AlbumData:
    """Analyze all photos and store results in AlbumData."""
    context = _build_context(album_data, settings)
    total = len(photos)

    all_results: list[dict] = []
    for i in range(0, total, BATCH_SIZE):
        batch = photos[i : i + BATCH_SIZE]
        results = analyze_batch(batch, context, client, i, total)
        all_results.extend(results)

    # Store results in album data
    for photo, result in zip(photos, all_results):
        fname = photo.path.name
        album_data.photo_dates[fname] = result["estimated_date"]
        album_data.photo_reasoning[fname] = result["reasoning"]
        album_data.photo_confidence[fname] = result["confidence"]
        album_data.photo_groups[fname] = result.get("group", 0)

    album_data.save()
    return album_data

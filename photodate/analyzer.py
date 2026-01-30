import base64
import json
from datetime import date
from io import BytesIO

import anthropic
from PIL import Image

from .models import DateEstimate, PhotoInfo

MAX_IMAGE_SIZE = 1024
BATCH_SIZE = 10


def _resize_and_encode(path) -> tuple[str, str]:
    """Resize image to max MAX_IMAGE_SIZE px and return base64 + media type."""
    img = Image.open(path)
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

    # Convert to RGB if needed (handles RGBA, palette, etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


PROMPT_TEMPLATE = """\
Je analyseert ingescande foto's uit een fysiek fotoalbum.

Albumcontext: "{context}"

Dit zijn foto's {start} t/m {end} van {total} in het album, in albumvolgorde.

Schat voor elke foto de datum waarop de foto oorspronkelijk is gemaakt. Let op:
- Kledingstijlen, kapsels
- Auto's, technologie, apparaten
- Omgeving, seizoenaanwijzingen (vegetatie, licht, weer)
- De albumcontext hierboven
- Foto's staan in chronologische volgorde binnen het album

Als je alleen een seizoen of jaar kunt schatten, gebruik dan het midden van die
periode (bijv. 15 juli voor "zomer 1987").

Antwoord in JSON (geen markdown codeblock):
[{{"photo_index": N, "estimated_date": "YYYY-MM-DD", \
"confidence": "high|medium|low", "reasoning": "..."}}]
"""


def analyze_batch(
    photos: list[PhotoInfo],
    album_context: str,
    client: anthropic.Anthropic,
    start_offset: int = 0,
    total: int = 0,
) -> list[DateEstimate]:
    """Send a batch of photos to Claude for date estimation."""
    content: list[dict] = []

    for photo in photos:
        b64, media_type = _resize_and_encode(photo.path)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        })
        content.append({
            "type": "text",
            "text": f"Foto {photo.index + 1}: {photo.path.name}",
        })

    prompt = PROMPT_TEMPLATE.format(
        context=album_context,
        start=start_offset + 1,
        end=start_offset + len(photos),
        total=total or len(photos),
    )
    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )

    response_text = response.content[0].text
    # Strip markdown code fences if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    results = json.loads(response_text)

    estimates: list[DateEstimate] = []
    for r in results:
        estimates.append(DateEstimate(
            date=date.fromisoformat(r["estimated_date"]),
            confidence=r["confidence"],
            reasoning=r["reasoning"],
        ))

    return estimates


def analyze_album(
    photos: list[PhotoInfo],
    album_context: str,
    client: anthropic.Anthropic,
) -> list[PhotoInfo]:
    """Analyze all photos in an album in batches."""
    total = len(photos)

    for i in range(0, total, BATCH_SIZE):
        batch = photos[i : i + BATCH_SIZE]
        estimates = analyze_batch(batch, album_context, client, i, total)

        for photo, estimate in zip(batch, estimates):
            photo.estimate = estimate

    return photos

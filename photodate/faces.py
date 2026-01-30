"""Face detection and clustering using face_recognition library."""
import base64
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image

# Tolerance for face matching (lower = stricter)
FACE_TOLERANCE = 0.6
# Resize for faster processing
MAX_FACE_SIZE = 800


def _load_image(path: Path) -> np.ndarray:
    """Load and resize image for face detection."""
    img = Image.open(path)
    img.thumbnail((MAX_FACE_SIZE, MAX_FACE_SIZE))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def detect_faces_in_album(photo_paths: list[Path]) -> dict:
    """Detect and cluster faces across all photos in an album.

    Returns:
        {
            "clusters": [
                {
                    "id": "face_0",
                    "photo_indices": [0, 3, 7],  # which photos this face appears in
                    "sample_crops": ["base64...", ...],  # up to 3 sample face crops
                }
            ],
            "per_photo": {
                0: ["face_0", "face_2"],  # face cluster IDs found in photo 0
                3: ["face_0"],
            }
        }
    """
    all_encodings: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []

    # Step 1: Detect faces in each photo
    for idx, path in enumerate(photo_paths):
        try:
            img = _load_image(path)
            locations = face_recognition.face_locations(img, model="hog")
            encodings = face_recognition.face_encodings(img, locations)
            for enc, loc in zip(encodings, locations):
                all_encodings.append((idx, enc, loc))
        except Exception:
            continue

    if not all_encodings:
        return {"clusters": [], "per_photo": {}}

    # Step 2: Cluster faces by similarity
    clusters: list[list[int]] = []  # Each cluster is a list of indices into all_encodings
    assigned = [False] * len(all_encodings)

    for i in range(len(all_encodings)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(all_encodings)):
            if assigned[j]:
                continue
            dist = face_recognition.face_distance(
                [all_encodings[i][1]], all_encodings[j][1]
            )[0]
            if dist < FACE_TOLERANCE:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    # Step 3: Build result
    result_clusters = []
    per_photo: dict[int, list[str]] = defaultdict(list)

    for cluster_idx, member_indices in enumerate(clusters):
        cluster_id = f"face_{cluster_idx}"
        photo_indices = sorted(set(all_encodings[m][0] for m in member_indices))

        # Get up to 3 sample face crops as base64
        sample_crops = []
        for m in member_indices[:3]:
            photo_idx, _, (top, right, bottom, left) = all_encodings[m]
            try:
                img = Image.open(photo_paths[photo_idx])
                # Scale location back if image was resized
                orig_w, orig_h = img.size
                scale = max(orig_w, orig_h) / MAX_FACE_SIZE
                if scale > 1:
                    top = int(top * scale)
                    right = int(right * scale)
                    bottom = int(bottom * scale)
                    left = int(left * scale)
                # Add some padding
                pad = int((bottom - top) * 0.2)
                crop = img.crop((
                    max(0, left - pad),
                    max(0, top - pad),
                    min(orig_w, right + pad),
                    min(orig_h, bottom + pad),
                ))
                crop.thumbnail((100, 100))
                buf = BytesIO()
                crop.save(buf, format="JPEG", quality=80)
                sample_crops.append(base64.b64encode(buf.getvalue()).decode())
            except Exception:
                continue

        result_clusters.append({
            "id": cluster_id,
            "photo_indices": photo_indices,
            "sample_crops": sample_crops,
            "count": len(photo_indices),
        })

        for pi in photo_indices:
            per_photo[pi].append(cluster_id)

    # Sort clusters by frequency (most common first)
    result_clusters.sort(key=lambda c: c["count"], reverse=True)

    return {
        "clusters": result_clusters,
        "per_photo": dict(per_photo),
    }

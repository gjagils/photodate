"""Face detection and clustering using face_recognition library."""
import base64
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Optional

import face_recognition
import numpy as np
from PIL import Image

FACE_TOLERANCE = 0.55  # Slightly stricter than 0.6 to reduce false matches
MAX_FACE_SIZE = 800


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    img.thumbnail((MAX_FACE_SIZE, MAX_FACE_SIZE))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def detect_faces_in_album(
    photo_paths: list[Path],
    global_profiles: Optional[list[dict]] = None,
) -> dict:
    """Detect and cluster faces across all photos in an album.

    Uses centroid-based clustering: each new face is compared to the running
    mean encoding of existing clusters, giving better accuracy than greedy
    first-match.

    Args:
        photo_paths: List of photo file paths in album order.
        global_profiles: Optional list of {"name": str, "encoding": list[float]}
            from GlobalSettings. Matching clusters are auto-labeled.

    Returns:
        {
            "clusters": [
                {
                    "id": "face_0",
                    "photo_indices": [0, 3, 7],
                    "sample_crops": ["base64...", ...],
                    "count": N,
                    "centroid": [...],   # 128-dim encoding as list, for profile saving
                    "auto_label": "",    # from global_profiles if matched
                }
            ],
            "per_photo": {
                0: ["face_0", "face_2"],
            }
        }
    """
    all_encodings: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []

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

    # Centroid-based clustering: compare each new face to the running mean
    # of each cluster, assign to nearest within tolerance, else start new cluster
    cluster_centroids: list[np.ndarray] = []
    cluster_members: list[list[int]] = []  # indices into all_encodings

    for i, (_, enc, _) in enumerate(all_encodings):
        best_cluster = -1
        best_dist = FACE_TOLERANCE

        for c_idx, centroid in enumerate(cluster_centroids):
            dist = face_recognition.face_distance([centroid], enc)[0]
            if dist < best_dist:
                best_dist = dist
                best_cluster = c_idx

        if best_cluster >= 0:
            cluster_members[best_cluster].append(i)
            n = len(cluster_members[best_cluster])
            cluster_centroids[best_cluster] = (
                cluster_centroids[best_cluster] * (n - 1) + enc
            ) / n
        else:
            cluster_centroids.append(enc.copy())
            cluster_members.append([i])

    # Build result
    result_clusters = []
    per_photo: dict[int, list[str]] = defaultdict(list)

    for c_idx, (centroid, member_indices) in enumerate(
        zip(cluster_centroids, cluster_members)
    ):
        cluster_id = f"face_{c_idx}"
        photo_indices = sorted(set(all_encodings[m][0] for m in member_indices))

        # Try to auto-label from global profiles
        auto_label = ""
        if global_profiles:
            for profile in global_profiles:
                dist = face_recognition.face_distance(
                    [np.array(profile["encoding"])], centroid
                )[0]
                if dist < FACE_TOLERANCE:
                    auto_label = profile["name"]
                    break

        # Collect up to 3 sample face crops
        sample_crops = []
        for m in member_indices[:3]:
            photo_idx, _, (top, right, bottom, left) = all_encodings[m]
            try:
                img = Image.open(photo_paths[photo_idx])
                orig_w, orig_h = img.size
                scale = max(orig_w, orig_h) / MAX_FACE_SIZE
                if scale > 1:
                    top = int(top * scale)
                    right = int(right * scale)
                    bottom = int(bottom * scale)
                    left = int(left * scale)
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
            "centroid": centroid.tolist(),
            "auto_label": auto_label,
        })

        for pi in photo_indices:
            per_photo[pi].append(cluster_id)

    result_clusters.sort(key=lambda c: c["count"], reverse=True)

    return {
        "clusters": result_clusters,
        "per_photo": dict(per_photo),
    }

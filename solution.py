import base64
import html
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_FACES_DIR = Path("raw_faces")
REFERENCE_DIR = Path("reference_identities")
ENHANCED_DIR = Path("enhanced_faces")
REPORT_HTML_OUT = Path("enhancement_report.html")
METRICS_JSON_OUT = Path("evaluation_metrics.json")

TARGET_SIZE = (240, 240)
ENHANCED_DIR.mkdir(exist_ok=True)


_FACE_MESH = None
_FACE_MESH_UNAVAILABLE = False


# ---------------------------------------------------------------------------
# STAGE 1 -- DENOISE
# ---------------------------------------------------------------------------

def stage1_denoise(img: np.ndarray) -> np.ndarray:
    """
    cv2.fastNlMeansDenoisingColored with h=8, hColor=8, templateWindowSize=7, searchWindowSize=21
    TODO: one line
    """
    return cv2.fastNlMeansDenoisingColored(
        img, None, h=8, hColor=8, templateWindowSize=7, searchWindowSize=21
    )


# ---------------------------------------------------------------------------
# STAGE 2 -- CLAHE
# ---------------------------------------------------------------------------

def stage2_clahe(img: np.ndarray) -> np.ndarray:
    """
    Convert to LAB. Apply CLAHE (clipLimit=3.5, tileGridSize=(4,4)) to L channel. Merge + convert back.
    TODO: implement
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    l_chan = clahe.apply(l_chan)
    merged = cv2.merge((l_chan, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# STAGE 3 -- MULTI-STEP UPSCALE
# ---------------------------------------------------------------------------

def unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """
    blurred = GaussianBlur(img, sigma)
    result  = img + strength * (img - blurred)
    Clip to 0-255.
    TODO: implement with cv2.GaussianBlur + cv2.addWeighted
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)


def stage3_upscale(img: np.ndarray) -> np.ndarray:
    """
    If short side < 64px: 2x LANCZOS4 -> unsharp(1.0, 1.6) -> 2x LANCZOS4 -> resize to TARGET_SIZE.
    Otherwise: direct resize to TARGET_SIZE LANCZOS4.
    TODO: implement
    """
    if min(img.shape[:2]) < 64:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        img = unsharp_mask(img, sigma=1.0, strength=1.6)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# STAGE 4 -- ZONE SHARPENING
# ---------------------------------------------------------------------------

def _get_face_mesh():
    global _FACE_MESH, _FACE_MESH_UNAVAILABLE

    if _FACE_MESH_UNAVAILABLE:
        return None
    if _FACE_MESH is not None:
        return _FACE_MESH

    try:
        import mediapipe as mp
    except ImportError:
        _FACE_MESH_UNAVAILABLE = True
        return None

    try:
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
    except Exception:
        _FACE_MESH_UNAVAILABLE = True
        return None

    return _FACE_MESH


def _landmark_to_point(landmark, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(landmark.x * width, 0, width - 1))
    y = int(np.clip(landmark.y * height, 0, height - 1))
    return x, y


def _focus_mask_from_facemesh(img: np.ndarray) -> Optional[np.ndarray]:
    face_mesh = _get_face_mesh()
    if face_mesh is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]

    left_eye_ids = [33, 133, 159, 145]
    right_eye_ids = [362, 263, 386, 374]
    nose_ids = [1, 4, 94, 195]

    points = np.array(
        [_landmark_to_point(landmarks[idx], w, h) for idx in left_eye_ids + right_eye_ids + nose_ids],
        dtype=np.int32,
    )
    if len(points) == 0:
        return None

    mask = np.zeros((h, w), dtype=np.float32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 1.0)

    for idx in left_eye_ids + right_eye_ids:
        cx, cy = _landmark_to_point(landmarks[idx], w, h)
        radius = max(4, int(min(h, w) * 0.04))
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)

    for idx in nose_ids:
        cx, cy = _landmark_to_point(landmarks[idx], w, h)
        radius = max(5, int(min(h, w) * 0.05))
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6.0, sigmaY=6.0)
    return np.clip(mask[:, :, None], 0.0, 1.0)


def stage4_zone_sharpen(img: np.ndarray) -> np.ndarray:
    """
    MediaPipe Face Mesh -> locate eye + nose region -> create mask.
    Apply unsharp(0.8, 2.0) to eye+nose zone.
    Apply unsharp(1.2, 1.3) to the rest.
    Blend using the mask.
    Fallback if no face found: unsharp(1.0, 1.5) uniformly.
    TODO: implement
    """
    mask = _focus_mask_from_facemesh(img)
    if mask is None:
        return unsharp_mask(img, sigma=1.0, strength=1.5)

    zone = unsharp_mask(img, sigma=0.8, strength=2.0).astype(np.float32)
    rest = unsharp_mask(img, sigma=1.2, strength=1.3).astype(np.float32)
    blended = zone * mask + rest * (1.0 - mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# FULL PIPELINE -- do not change this function
# ---------------------------------------------------------------------------

def enhance_face(img: np.ndarray) -> np.ndarray:
    """Run all 4 stages in order. Do not modify."""
    img = stage1_denoise(img)
    img = stage2_clahe(img)
    img = stage3_upscale(img)
    img = stage4_zone_sharpen(img)
    return img


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------

def sharpness(img: np.ndarray) -> float:
    """Laplacian variance. Higher = sharper. Convert to grayscale first.
    TODO: cv2.Laplacian(gray, cv2.CV_64F).var()
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_face_encoding(img: np.ndarray):
    """
    128-d face encoding. Return numpy array if face found, else None.
    Use number_of_times_to_upsample=2 for small faces.
    TODO: implement with face_recognition
    """
    try:
        import face_recognition as fr
    except ImportError:
        return None

    if img is None or img.size == 0:
        return None

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    locations = fr.face_locations(rgb, number_of_times_to_upsample=2, model="hog")
    if not locations:
        return None

    encodings = fr.face_encodings(
        rgb,
        known_face_locations=locations,
        num_jitters=1,
        model="small",
    )
    return encodings[0] if encodings else None


def _fallback_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return float(np.mean(numerator / (denominator + 1e-8)))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index between two images.
    Both resized to TARGET_SIZE before comparison. Convert to grayscale.
    Return float. Higher = more similar.
    TODO: from skimage.metrics import structural_similarity
    """
    a_resized = cv2.resize(a, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    b_resized = cv2.resize(b, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    gray_a = cv2.cvtColor(a_resized, cv2.COLOR_BGR2GRAY) if a_resized.ndim == 3 else a_resized
    gray_b = cv2.cvtColor(b_resized, cv2.COLOR_BGR2GRAY) if b_resized.ndim == 3 else b_resized

    if structural_similarity is not None:
        return float(structural_similarity(gray_a, gray_b, data_range=255))
    return _fallback_ssim(gray_a, gray_b)


# ---------------------------------------------------------------------------
# HTML A/B REPORT
# ---------------------------------------------------------------------------

def generate_ab_report(results: list, output_path: Path):
    """
    Self-contained HTML. No CDN.

    Summary header: overall accuracy improvement + sharpness gain.
    Grid: each row = original image | enhanced image | sharpness before/after | match before/after.
    Images embedded as base64.

    TODO: implement
    """
    total = len(results)
    acc_before = (sum(r["match_before"] for r in results) / total * 100.0) if total else 0.0
    acc_after = (sum(r["match_after"] for r in results) / total * 100.0) if total else 0.0
    sharp_before = float(np.mean([r["sharpness_before"] for r in results])) if results else 0.0
    sharp_after = float(np.mean([r["sharpness_after"] for r in results])) if results else 0.0

    rows = []
    for row in results:
        matched_identity = row["matched_identity"] if row["matched_identity"] is not None else "-"
        raw_src = html.escape(f"raw_faces/{row['filename']}")
        enhanced_src = html.escape(f"enhanced_faces/{row['filename']}")
        rows.append(
            f"""
            <div class="card">
              <div class="header">{html.escape(row["filename"])}</div>
              <div class="images">
                <figure>
                  <img src="{raw_src}" alt="raw {html.escape(row["filename"])}" loading="lazy">
                  <figcaption>Original</figcaption>
                </figure>
                <figure>
                  <img src="{enhanced_src}" alt="enhanced {html.escape(row["filename"])}" loading="lazy">
                  <figcaption>Enhanced</figcaption>
                </figure>
              </div>
              <div class="stats">
                <div><strong>Sharpness:</strong> {row["sharpness_before"]} -> {row["sharpness_after"]}</div>
                <div><strong>SSIM:</strong> {row["ssim_improvement"]}</div>
                <div><strong>Match:</strong> {row["match_before"]} -> {row["match_after"]}</div>
                <div><strong>Identity:</strong> {html.escape(str(matched_identity))}</div>
              </div>
            </div>
            """
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CCTV Face Enhancement Report</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #111827;
      color: #f3f4f6;
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .summary {{
      background: #1f2937;
      border: 1px solid #374151;
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 24px;
    }}
    .summary h1 {{
      margin: 0 0 12px;
      font-size: 28px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .summary-item {{
      background: #0f172a;
      border-radius: 12px;
      padding: 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #1f2937;
      border: 1px solid #374151;
      border-radius: 16px;
      overflow: hidden;
    }}
    .header {{
      padding: 14px 16px;
      font-weight: 700;
      border-bottom: 1px solid #374151;
    }}
    .images {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 16px;
    }}
    figure {{
      margin: 0;
      text-align: center;
    }}
    img {{
      width: 100%;
      max-width: 240px;
      height: auto;
      border-radius: 10px;
      border: 1px solid #4b5563;
      background: #000;
    }}
    figcaption {{
      margin-top: 8px;
      font-size: 14px;
      color: #cbd5e1;
    }}
    .stats {{
      padding: 0 16px 16px;
      display: grid;
      gap: 8px;
      color: #d1d5db;
    }}
    strong {{
      color: #ffffff;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="summary">
      <h1>Low-Resolution CCTV Face Enhancement Report</h1>
      <div class="summary-grid">
        <div class="summary-item"><strong>Total Faces</strong><br>{total}</div>
        <div class="summary-item"><strong>Recognition</strong><br>{acc_before:.1f}% -> {acc_after:.1f}%</div>
        <div class="summary-item"><strong>Sharpness</strong><br>{sharp_before:.2f} -> {sharp_after:.2f}</div>
        <div class="summary-item"><strong>Sharpness Gain</strong><br>{(sharp_after - sharp_before):.2f}</div>
      </div>
    </section>
    <section class="grid">
      {''.join(rows)}
    </section>
  </div>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    # Load reference encodings for evaluation
    reference_encodings = {}
    for ref in sorted(REFERENCE_DIR.glob("*")):
        if ref.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        img = cv2.imread(str(ref))
        if img is None:
            continue
        enc = get_face_encoding(img)
        if enc is not None:
            reference_encodings[ref.stem] = enc
            print(f"  Reference: {ref.stem}")
        else:
            print(f"  WARNING: no face in {ref.name}")

    print(f"Loaded {len(reference_encodings)} reference identities")

    face_paths = sorted(RAW_FACES_DIR.glob("*.jpg")) + sorted(RAW_FACES_DIR.glob("*.png"))
    print(f"Processing {len(face_paths)} face crops ...")

    results = []

    for fp in face_paths:
        raw = cv2.imread(str(fp))
        if raw is None:
            continue

        enhanced = enhance_face(raw.copy())
        cv2.imwrite(str(ENHANCED_DIR / fp.name), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

        sharp_b = sharpness(raw)
        sharp_a = sharpness(enhanced)
        ssim_g = ssim_score(cv2.resize(raw, TARGET_SIZE), enhanced)

        enc_raw = get_face_encoding(raw)
        enc_enh = get_face_encoding(enhanced)

        match_b = False
        match_a = False
        mid = None

        if reference_encodings:
            import face_recognition as fr

            refs = list(reference_encodings.values())
            names = list(reference_encodings.keys())
            if enc_raw is not None:
                match_b = any(fr.compare_faces(refs, enc_raw, tolerance=0.60))
            if enc_enh is not None:
                hits = fr.compare_faces(refs, enc_enh, tolerance=0.60)
                match_a = any(hits)
                if match_a:
                    mid = names[hits.index(True)]

        # Encode for report
        _, rb = cv2.imencode(".jpg", cv2.resize(raw, TARGET_SIZE), [cv2.IMWRITE_JPEG_QUALITY, 82])
        _, eb = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 82])

        results.append({
            "filename": fp.name,
            "original_size_px": list(raw.shape[:2]),
            "enhanced_size_px": list(enhanced.shape[:2]),
            "sharpness_before": round(sharp_b, 2),
            "sharpness_after": round(sharp_a, 2),
            "ssim_improvement": round(ssim_g, 4),
            "match_before": match_b,
            "match_after": match_a,
            "matched_identity": mid,
            "raw_b64": base64.b64encode(rb).decode(),
            "enhanced_b64": base64.b64encode(eb).decode(),
        })
        print(f"  {fp.name}: sharp {sharp_b:.1f}->{sharp_a:.1f}  match {match_b}->{match_a}")

    n = len(results)
    t_s = round(time.time() - t_start, 2)

    metrics = {
        "source": "p4_face_enhancement",
        "total_faces_processed": n,
        "processing_time_sec": t_s,
        "pipeline_stages_applied": ["denoise", "clahe", "upscale_multistep", "zone_sharpen"],
        "recognition_accuracy_before_pct": round(sum(r["match_before"] for r in results) / n * 100, 1) if n else 0.0,
        "recognition_accuracy_after_pct": round(sum(r["match_after"] for r in results) / n * 100, 1) if n else 0.0,
        "avg_sharpness_before": round(float(np.mean([r["sharpness_before"] for r in results])), 2) if results else 0.0,
        "avg_sharpness_after": round(float(np.mean([r["sharpness_after"] for r in results])), 2) if results else 0.0,
        "avg_ssim_improvement": round(float(np.mean([r["ssim_improvement"] for r in results])), 4) if results else 0.0,
        "per_face": [{k: v for k, v in r.items() if k not in ["raw_b64", "enhanced_b64"]} for r in results],
    }

    with open(METRICS_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    generate_ab_report(results, REPORT_HTML_OUT)

    print()
    print("=" * 55)
    print(f"  Done in {t_s}s  for {n} faces")
    print(f"  Recognition:  {metrics['recognition_accuracy_before_pct']}%  ->  {metrics['recognition_accuracy_after_pct']}%")
    print(f"  Sharpness:    {metrics['avg_sharpness_before']}  ->  {metrics['avg_sharpness_after']}")
    print(f"  Enhanced  -> {ENHANCED_DIR}/")
    print(f"  Report    -> {REPORT_HTML_OUT}")
    print(f"  Metrics   -> {METRICS_JSON_OUT}")
    print("=" * 55)

# RedZone Sentinel — OpenCV Evidence Engine
### Your build guide (Module: Field Image → Change Evidence)

This is your working guide for the computer-vision part of RedZone Sentinel. You own everything between "two field photos" and "evidence the scoring engine can trust." Build it in the order below — each stage produces something you can actually run and show, and each one explains the *why*, not just the *how*, so you can defend it to judges.

---

## 0. The pipeline you're building

```
FIELD PHOTOS
     │
     ▼
IMAGE QUALITY CHECK      "Is this image usable?"
     │
     ▼
IMAGE ALIGNMENT          "Are we looking at the same place?"
     │
     ▼
CHANGE DETECTION         "What actually changed?"
     │
     ▼
NOISE / FALSE-CHANGE REMOVAL
     │
     ▼
CHANGE QUANTIFICATION    "How much, and how confident are we?"
     │
     ▼
EVIDENCE REPORT (images + JSON)   →  hands off to the scoring engine
```

**Golden rule for the demo:** never say "AI detected a disaster." Say "the vision module detected land-use change with X% confidence"; the scoring engine decides what that means. This separation is what makes the project defensible under judge questioning.

---

## 1. Project setup

```
RedZone_Sentinel/
├── data/
│   ├── before/
│   └── after/
├── output/
├── src/
│   ├── image_loader.py
│   ├── quality_check.py
│   ├── alignment.py
│   ├── change_detection.py
│   ├── evidence.py
│   └── main.py
└── requirements.txt
```

Install dependencies:

```bash
pip install opencv-python numpy matplotlib
```

Sanity check (`src/test_opencv.py`):

```python
import cv2
print("OpenCV version:", cv2.__version__)
```

**Test images:** for your first controlled test, don't grab random internet photos — take (or fake) two photos of the same static scene where you *know* what changed: e.g., empty ground → same ground with a structure added. This gives you ground truth to validate against before touching real field data.

---

## 2. Stage 1 — Load and inspect images

`src/image_loader.py`

```python
import cv2

def load_image(path: str):
    """Load an image and fail loudly if it's missing or unreadable."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def describe(img, label: str):
    h, w, c = img.shape
    print(f"{label}: {w}x{h}, {c} channels")

if __name__ == "__main__":
    before = load_image("data/before/before.jpg")
    after = load_image("data/after/after.jpg")
    describe(before, "Before")
    describe(after, "After")
```

**Concept to remember:** OpenCV stores color images as **BGR**, not RGB. If you ever hand an image to `matplotlib` for display, convert it first with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` or your colors will look wrong.

---

## 3. Stage 2 — Image quality check

Before trusting an image, ask two questions: is it sharp, and is it a reasonable size?

`src/quality_check.py`

```python
import cv2
import numpy as np

def sharpness_score(img) -> float:
    """Variance of the Laplacian — higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def quality_report(img, min_sharpness=100.0, min_dim=400) -> dict:
    h, w = img.shape[:2]
    score = sharpness_score(img)
    issues = []

    if score < min_sharpness:
        issues.append("too blurry")
    if min(h, w) < min_dim:
        issues.append("resolution too low")

    return {
        "sharpness": round(score, 1),
        "resolution": f"{w}x{h}",
        "usable": len(issues) == 0,
        "issues": issues,
    }

if __name__ == "__main__":
    from image_loader import load_image
    img = load_image("data/before/before.jpg")
    print(quality_report(img))
```

**Why this matters for the demo:** instead of your system confidently claiming change on a blurry photo, it can say `"Evidence confidence: LOW — image quality insufficient"`. That's a much stronger engineering answer than pretending every photo is perfect.

`min_sharpness=100.0` is a starting guess — you'll tune this once you see real Laplacian variance numbers from your own test photos.

---

## 4. Stage 3 — Image alignment (the important one)

If the camera moved even slightly between "before" and "after," a naive pixel diff will scream "CHANGE!" everywhere, even if nothing actually changed. So you align the two images first — this is called **image registration**.

**Method: ORB features + homography**

```
BEFORE + AFTER
      │
Find matching keypoints (ORB)
      │
Filter to "good" matches
      │
Estimate homography (RANSAC)
      │
Warp AFTER onto BEFORE's frame
      │
Now they're geometrically aligned
```

`src/alignment.py`

```python
import cv2
import numpy as np

def align_images(before, after, max_features=2000, good_match_ratio=0.75):
    """
    Aligns `after` onto `before`'s coordinate frame using ORB + homography.
    Returns (aligned_after, diagnostics_dict).
    """
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(gray_before, None)
    kp2, des2 = orb.detectAndCompute(gray_after, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        raise ValueError("Not enough features found to align images.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test — keep only distinctive, reliable matches
    good_matches = []
    for m, n in raw_matches:
        if m.distance < good_match_ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        raise ValueError("Not enough good matches to compute alignment.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0

    h, w = before.shape[:2]
    aligned_after = cv2.warpPerspective(after, H, (w, h))

    diagnostics = {
        "feature_matches": len(raw_matches),
        "good_matches": len(good_matches),
        "inliers": inliers,
        "alignment_confidence": round(100 * inliers / max(len(good_matches), 1), 1),
    }
    return aligned_after, diagnostics

if __name__ == "__main__":
    from image_loader import load_image
    before = load_image("data/before/before.jpg")
    after = load_image("data/after/after.jpg")
    aligned, diag = align_images(before, after)
    print(diag)
    cv2.imwrite("output/aligned_after.jpg", aligned)
```

**Judge-demo moment:** show `Before | After | Aligned After` side by side, and print the diagnostics table. Numbers like `Good matches: 127, Inliers: 103, Alignment confidence: 91%` make it obvious your system did real work, not "we used OpenCV."

If alignment confidence comes back very low (e.g., under ~40%), that itself is useful information — flag the pair as "unreliable, needs re-shoot" rather than forcing a comparison.

---

## 5. Stage 4 — Change detection (multi-signal)

Don't rely on a single `cv2.absdiff()`. Lighting, shadows, and weather will drown you in false positives. Combine three signals instead.

`src/change_detection.py`

```python
import cv2
import numpy as np

def compute_change_mask(before, aligned_after, blur_ksize=5):
    """
    Combines intensity, color (Lab), and structural signals into
    a single binary change mask.
    """
    # Reduce sensor noise before comparing
    b = cv2.GaussianBlur(before, (blur_ksize, blur_ksize), 0)
    a = cv2.GaussianBlur(aligned_after, (blur_ksize, blur_ksize), 0)

    # Signal 1: grayscale intensity difference
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    intensity_diff = cv2.absdiff(gray_b, gray_a)

    # Signal 2: color difference in Lab space (more perceptually robust
    # to brightness swings than raw BGR)
    lab_b = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
    lab_a = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    color_diff = cv2.absdiff(lab_b, lab_a)
    color_diff = cv2.cvtColor(color_diff, cv2.COLOR_LAB2BGR)
    color_diff = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)

    # Signal 3: structural difference via local standard deviation
    # (captures texture change, e.g. cleared land vs. dense trees)
    def local_std(gray, k=9):
        mean = cv2.blur(gray.astype(np.float32), (k, k))
        sq_mean = cv2.blur((gray.astype(np.float32)) ** 2, (k, k))
        return np.sqrt(np.maximum(sq_mean - mean ** 2, 0))

    struct_diff = cv2.absdiff(local_std(gray_b), local_std(gray_a))
    struct_diff = cv2.normalize(struct_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Combine signals (weighted — tune these on your own test set)
    combined = cv2.addWeighted(intensity_diff, 0.4, color_diff, 0.35, 0)
    combined = cv2.addWeighted(combined, 1.0, struct_diff, 0.25, 0)

    # Threshold to binary mask (Otsu picks the cut automatically)
    _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask, {
        "intensity_diff": intensity_diff,
        "color_diff": color_diff,
        "struct_diff": struct_diff,
        "combined": combined,
    }
```

---

## 6. Stage 5 — Clean the mask (remove false-change noise)

Raw masks are speckled with tiny false positives. Clean them with morphology, then keep only regions big enough to matter.

Add to `src/change_detection.py`:

```python
def clean_mask(mask, min_area=500, kernel_size=5):
    """Morphological cleanup + small-region removal."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove speckle
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)  # fill small gaps

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros_like(cleaned)
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append({"area_px": int(area), "bbox": (x, y, w, h)})

    regions.sort(key=lambda r: r["area_px"], reverse=True)
    return final_mask, regions
```

`min_area=500` is a starting point in pixels — scale it relative to your image resolution (e.g., 0.02% of total image area) rather than hardcoding it once you move to real field photos.

---

## 7. Stage 6 — Quantify the change

```python
def change_summary(mask, regions):
    total_px = mask.shape[0] * mask.shape[1]
    changed_px = int(np.count_nonzero(mask))
    pct = round(100 * changed_px / total_px, 2)

    return {
        "total_area_px": total_px,
        "changed_area_px": changed_px,
        "changed_area_pct": pct,
        "significant_regions": len(regions),
        "top_regions": [
            {"area_px": r["area_px"], "relative_pct": round(100 * r["area_px"] / total_px, 2)}
            for r in regions[:5]
        ],
    }
```

Combine this with alignment confidence and image quality into one **evidence confidence** score — label it explicitly as a confidence/evidence metric, not a hazard probability:

```python
def evidence_confidence(alignment_conf, quality_ok_before, quality_ok_after, num_regions):
    score = alignment_conf
    if not quality_ok_before or not quality_ok_after:
        score *= 0.6
    if num_regions == 0:
        score *= 0.7  # no coherent region = weaker evidence even if % is nonzero
    if score >= 80:
        label = "HIGH"
    elif score >= 55:
        label = "MEDIUM"
    else:
        label = "LOW"
    return round(score, 1), label
```

---

## 8. Stage 7 — Build the evidence report

`src/evidence.py`

```python
import cv2
import numpy as np

def draw_overlay(before, mask, color=(0, 0, 255), alpha=0.45):
    """Highlight changed regions in red on top of the before image."""
    overlay = before.copy()
    colored = np.zeros_like(before)
    colored[:] = color
    mask_3ch = cv2.merge([mask, mask, mask])
    highlighted = np.where(mask_3ch > 0, colored, before)
    return cv2.addWeighted(before, 1 - alpha, highlighted, alpha, 0)

def build_evidence_image(before, aligned_after, mask, out_path):
    """Before | After | Overlay, side by side, saved as one image."""
    overlay = draw_overlay(before, mask)
    h, w = before.shape[:2]

    def label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    panel = np.hstack([label(before, "BEFORE"), label(aligned_after, "AFTER"), label(overlay, "CHANGE")])
    cv2.imwrite(out_path, panel)
    return out_path
```

Save this as `output/evidence_report.jpg` — this single image is your best asset for the SIH demo. It shows the judges exactly what the algorithm saw, with no need to trust a black box.

---

## 9. Stage 8 — Wire it all together

`src/main.py`

```python
import json
from image_loader import load_image
from quality_check import quality_report
from alignment import align_images
from change_detection import compute_change_mask, clean_mask, change_summary
from evidence import build_evidence_image
from confidence import evidence_confidence  # or inline it from stage 6

def run_pipeline(before_path, after_path, out_dir="output"):
    before = load_image(before_path)
    after = load_image(after_path)

    q_before = quality_report(before)
    q_after = quality_report(after)

    aligned_after, align_diag = align_images(before, after)

    raw_mask, _signals = compute_change_mask(before, aligned_after)
    final_mask, regions = clean_mask(raw_mask)
    summary = change_summary(final_mask, regions)

    conf_score, conf_label = evidence_confidence(
        align_diag["alignment_confidence"],
        q_before["usable"], q_after["usable"],
        summary["significant_regions"],
    )

    evidence_path = build_evidence_image(before, aligned_after, final_mask, f"{out_dir}/evidence_report.jpg")

    result = {
        "quality": {"before": q_before, "after": q_after},
        "alignment": align_diag,
        "change": summary,
        "evidence_confidence": {"score": conf_score, "label": conf_label},
        "evidence_image": evidence_path,
    }

    with open(f"{out_dir}/result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    result = run_pipeline("data/before/before.jpg", "data/after/after.jpg")
    print(json.dumps(result, indent=2))
```

This `result.json` is exactly what you hand to the FastAPI layer / scoring engine — no image processing leaks past this boundary.

---

## 10. Suggested build order (don't skip steps)

| # | Task | Output you should see |
|---|------|------------------------|
| 1 | Install OpenCV, run `test_opencv.py` | version string prints |
| 2 | `image_loader.py` on real test images | shapes print correctly |
| 3 | `quality_check.py` | sharpness score + usable/not |
| 4 | `alignment.py` on a deliberately shifted photo pair | alignment confidence > 0, visibly aligned output |
| 5 | `change_detection.py` on a *no-change* pair (same scene, camera moved) | change % should be low after alignment — this is your false-positive test |
| 6 | `change_detection.py` on a *controlled-change* pair (you added/removed something) | change % should be meaningfully higher, region(s) detected |
| 7 | `evidence.py` | `evidence_report.jpg` looks presentable |
| 8 | `main.py` end-to-end | clean `result.json` |
| 9 | Tune thresholds (`min_area`, `min_sharpness`, blend weights) against your test set | fewer false positives, no missed real changes |
| 10 | Hand `result.json` to your teammate's FastAPI/scoring layer | integration test |

**Two test cases you must be able to demo:**
- **False-change resistance:** same location, camera slightly moved / different lighting → system reports low change after alignment.
- **Real change detection:** controlled before/after with an actual change → system reports the change, region count, and a "why flagged" checklist.

---

## 11. What to say to judges (and what not to say)

- ❌ "Our AI detects disasters."
- ✅ "Our vision module detects and quantifies land-use change between two field photos, with a confidence score based on image quality and alignment reliability. The scoring engine interprets that evidence alongside other signals to decide priority."

This framing — *evidence engine, not decision engine* — is also your best defense if a judge pushes on false positives: your answer is "we surface confidence, we don't claim certainty," which is a genuinely correct engineering position, not a dodge.

---

## Next steps once this works

- Swap `min_area`/thresholds for values tuned on real field photos, not just your controlled test.
- Add a "why flagged" text generator that reads `result.json` and prints the checklist (% change, region count, alignment confidence) as bullet points for the dashboard.
- Consider drought/vegetation-specific handling later (e.g., NDVI-like ratios) only if your dataset needs it — don't over-build before the basic pipeline is solid.

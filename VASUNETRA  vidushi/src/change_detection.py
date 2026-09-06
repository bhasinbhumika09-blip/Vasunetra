import cv2
import numpy as np


def compute_change_mask(before, aligned_after, valid_mask=None, blur_ksize=5):
    """
    Combines intensity, color (Lab), and structural signals into
    a single binary change mask. If valid_mask is given (255 = real pixel,
    0 = empty border introduced by alignment warp), those border pixels
    are excluded so warp artifacts never get reported as change.
    """
    b = cv2.GaussianBlur(before, (blur_ksize, blur_ksize), 0)
    a = cv2.GaussianBlur(aligned_after, (blur_ksize, blur_ksize), 0)

    # Signal 1: grayscale intensity difference
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    intensity_diff = cv2.absdiff(gray_b, gray_a)

    # Signal 2: color difference in Lab space (more robust to brightness
    # swings than raw BGR). Average the per-channel absolute difference —
    # do NOT convert the difference back through Lab2BGR, since a diff
    # image isn't a valid Lab image and that produces meaningless noise.
    lab_b = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
    lab_a = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    color_diff_raw = cv2.absdiff(lab_b, lab_a)
    color_diff = np.mean(color_diff_raw, axis=2).astype(np.uint8)

    # Signal 3: structural difference via local standard deviation
    def local_std(gray, k=9):
        gray_f = gray.astype(np.float32)
        mean = cv2.blur(gray_f, (k, k))
        sq_mean = cv2.blur(gray_f ** 2, (k, k))
        return np.sqrt(np.maximum(sq_mean - mean ** 2, 0))

    struct_diff = cv2.absdiff(local_std(gray_b), local_std(gray_a))
    struct_diff = cv2.normalize(struct_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    combined = cv2.addWeighted(intensity_diff, 0.4, color_diff, 0.35, 0)
    combined = cv2.addWeighted(combined, 1.0, struct_diff, 0.25, 0)

    if valid_mask is not None:
        # Zero out anything outside the real (non-border) image area
        # BEFORE thresholding, so it can't skew where Otsu sets the cut.
        combined = cv2.bitwise_and(combined, combined, mask=valid_mask)

    _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if valid_mask is not None:
        mask = cv2.bitwise_and(mask, valid_mask)

    return mask, {
        "intensity_diff": intensity_diff,
        "color_diff": color_diff,
        "struct_diff": struct_diff,
        "combined": combined,
    }


def clean_mask(mask, min_area=500, kernel_size=5):
    """Morphological cleanup + small-region removal."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

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

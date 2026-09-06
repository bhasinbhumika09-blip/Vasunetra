import cv2
import numpy as np


def draw_overlay(before, mask, regions=None, color=(0, 0, 255), alpha=0.5):
    """Highlight changed regions in bright red, with a solid outline and
    a labeled bounding box around each significant region, so the change
    is unmistakable even at a glance."""
    colored = np.zeros_like(before)
    colored[:] = color
    mask_3ch = cv2.merge([mask, mask, mask])
    highlighted = np.where(mask_3ch > 0, colored, before)
    overlay = cv2.addWeighted(before, 1 - alpha, highlighted, alpha, 0)

    # Solid outline around the exact changed region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)

    # Bright bounding box + area label for each significant region
    if regions:
        for r in regions:
            x, y, w, h = r["bbox"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(overlay, f'{r["area_px"]}px', (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return overlay


def build_evidence_image(before, aligned_after, mask, out_path, regions=None):
    """Before | After | Overlay, side by side, saved as one image."""
    overlay = draw_overlay(before, mask, regions=regions)
    h, w = before.shape[:2]

    def label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    panel = np.hstack([label(before, "BEFORE"), label(aligned_after, "AFTER"), label(overlay, "CHANGE")])
    cv2.imwrite(out_path, panel)
    return out_path

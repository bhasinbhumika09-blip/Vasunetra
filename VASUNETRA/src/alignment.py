import cv2
import numpy as np

def align_images(before, after, max_features=2000, good_match_ratio=0.75):
    """
    Aligns `after` onto `before`'s coordinate frame using ORB + homography.
    Returns (aligned_after, valid_mask, diagnostics_dict).

    valid_mask marks pixels that came from real image data after warping
    (255) vs. empty border area introduced by the warp itself (0) — this
    lets change_detection ignore fake "change" caused by alignment, not
    by anything that actually happened in the field.
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

    # Track which pixels are real vs. empty border introduced by the warp
    ones = np.full((after.shape[0], after.shape[1]), 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(ones, H, (w, h))
    # shrink it a little further so we also drop the unreliable edge band
    valid_mask = cv2.erode(valid_mask, np.ones((15, 15), np.uint8))

    diagnostics = {
        "feature_matches": len(raw_matches),
        "good_matches": len(good_matches),
        "inliers": inliers,
        "alignment_confidence": round(100 * inliers / max(len(good_matches), 1), 1),
    }
    return aligned_after, valid_mask, diagnostics

if __name__ == "__main__":
    from image_loader import load_image
    before = load_image("data/before/before.jpg")
    after = load_image("data/after/after.jpg")
    aligned, valid_mask, diag = align_images(before, after)
    print(diag)
    cv2.imwrite("output/aligned_after.jpg", aligned)

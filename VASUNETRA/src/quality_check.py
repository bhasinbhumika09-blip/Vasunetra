import cv2

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

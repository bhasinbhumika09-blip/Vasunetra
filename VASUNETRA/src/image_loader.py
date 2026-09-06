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

"""Hand skeleton rendering with PIL (no OpenCV dependency in the UI path)."""
import numpy as np
from PIL import Image, ImageDraw

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]
FINGER_COLORS = {
    "thumb": "#e07a5f", "index": "#3d9970", "middle": "#4361ee",
    "ring": "#b5179e", "pinky": "#f77f00", "palm": "#6c757d",
}


def _color(a, b):
    if a == 0 and b == 1 or 1 <= a <= 3:
        return FINGER_COLORS["thumb"]
    if a == 0 and b == 5 or 5 <= a <= 7:
        return FINGER_COLORS["index"]
    if a == 0 and b == 9 or 9 <= a <= 11:
        return FINGER_COLORS["middle"]
    if a == 0 and b == 13 or 13 <= a <= 15:
        return FINGER_COLORS["ring"]
    if a == 0 and b == 17 or 17 <= a <= 19:
        return FINGER_COLORS["pinky"]
    return FINGER_COLORS["palm"]


def render_skeleton(landmarks, size=420, background="#0f1117"):
    """Draw a (21,2) math-coords landmark array as a color-coded skeleton image."""
    pts = np.asarray(landmarks, dtype=float)[:, :2]
    span = pts.max(axis=0) - pts.min(axis=0)
    pad = 0.15 * max(span.max(), 1e-6)
    lo = pts.min(axis=0) - pad
    scale = (size - 1) / (span.max() + 2 * pad)
    px = (pts - lo) * scale
    px[:, 1] = size - 1 - px[:, 1]  # math -> image coords
    img = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(img)
    for a, b in CONNECTIONS:
        draw.line([tuple(px[a]), tuple(px[b])], fill=_color(a, b), width=5)
    for i, (x, y) in enumerate(px):
        r = 8 if i in (0, 4, 8, 12, 16, 20) else 5
        draw.ellipse([x - r, y - r, x + r, y + r], fill="#f8f9fa", outline="#212529")
    return img


def overlay_skeleton(pil_image, landmarks_px):
    """Overlay skeleton on a photo. landmarks_px are math-coords pixels (y up)."""
    img = pil_image.convert("RGB").copy()
    h = img.height
    px = np.asarray(landmarks_px, dtype=float).copy()
    px[:, 1] = h - px[:, 1]
    draw = ImageDraw.Draw(img)
    for a, b in CONNECTIONS:
        draw.line([tuple(px[a]), tuple(px[b])], fill="#00e676", width=4)
    for x, y in px:
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="#ffffff", outline="#111111")
    return img

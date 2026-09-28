"""Landmark normalization + feature extraction (shared by training and inference)."""
import numpy as np

WRIST, MIDDLE_MCP, INDEX_MCP, PINKY_MCP = 0, 9, 5, 17
TIPS = (4, 8, 12, 16, 20)
FINGER_MCP_TIP = ((5, 8), (9, 12), (13, 16), (17, 20))


def normalize(landmarks):
    """Canonicalize a (21,2+) landmark array: wrist at origin, wrist->middle-MCP
    aligned to +y, unit palm scale, thumb side forced to +x (handedness-invariant)."""
    pts = np.asarray(landmarks, dtype=float)[:, :2].copy()
    pts -= pts[WRIST]
    ref = pts[MIDDLE_MCP]
    scale = np.linalg.norm(ref)
    if scale < 1e-6:
        raise ValueError("Degenerate landmarks: wrist and middle MCP coincide")
    pts /= scale
    ref = pts[MIDDLE_MCP]
    ang = np.arctan2(ref[0], ref[1])  # rotate so ref points along +y
    c, s = np.cos(ang), np.sin(ang)
    pts = pts @ np.array([[c, -s], [s, c]]).T
    if pts[INDEX_MCP, 0] < pts[PINKY_MCP, 0]:  # mirror left hands to right
        pts[:, 0] *= -1
    return pts


def extract_features(landmarks):
    """56-dim feature vector: 42 normalized coords + 14 geometric descriptors."""
    pts = normalize(landmarks)
    tip_wrist = [np.linalg.norm(pts[t]) for t in TIPS]
    tip_mcp = [np.linalg.norm(pts[t] - pts[m]) for m, t in FINGER_MCP_TIP]
    gaps = [np.linalg.norm(pts[a] - pts[b]) for a, b in ((4, 8), (8, 12), (12, 16))]
    thumb_vec = list(pts[4] - pts[2])
    return np.concatenate([pts.ravel(), tip_wrist, tip_mcp, gaps, thumb_vec])

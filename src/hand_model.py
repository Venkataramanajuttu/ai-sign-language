"""Parametric 21-landmark hand model (MediaPipe topology) + ASL gesture templates.

Landmark order matches MediaPipe Hands:
0 wrist | 1-4 thumb (CMC,MCP,IP,TIP) | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
Coordinates: wrist at origin, fingers point +y, thumb side is +x. Palm width ~= 1.
"""
import math
from dataclasses import dataclass, field

import numpy as np

FINGER_MCPS = {"index": (0.30, 1.00), "middle": (0.10, 1.05),
               "ring": (-0.10, 1.00), "pinky": (-0.30, 0.90)}
FINGER_LENGTHS = {"index": (0.36, 0.22, 0.15), "middle": (0.40, 0.25, 0.17),
                  "ring": (0.36, 0.22, 0.15), "pinky": (0.26, 0.17, 0.12)}
THUMB_BASE = (0.32, 0.25)
THUMB_LENGTHS = (0.28, 0.22, 0.16)


@dataclass
class Gesture:
    """curl: 0=straight..1=fully curled per finger; splay: degrees; thumb_ang: deg from +x."""
    name: str
    display: str
    curls: tuple  # (index, middle, ring, pinky)
    thumb_ang: float
    thumb_curl: float
    splays: tuple = (4.0, 0.0, -4.0, -8.0)
    meaning: str = ""


GESTURES = [
    Gesture("A", "Letter A", (1, 1, 1, 1), 80, 0.35, meaning="Closed fist, thumb at side"),
    Gesture("B", "Letter B", (0.02, 0.02, 0.02, 0.04), 130, 0.50, (2, 0, -2, -5),
            "Flat hand, fingers up, thumb across palm"),
    Gesture("C", "Letter C", (0.5, 0.5, 0.5, 0.5), 20, 0.35, (3, 1, -1, -3),
            "Curved hand forming a C shape"),
    Gesture("D", "Letter D", (0.02, 0.85, 0.9, 0.9), 45, 0.55, (3, 0, 0, -3),
            "Index up, other fingers touch thumb"),
    Gesture("F", "Letter F", (0.65, 0.03, 0.03, 0.05), 35, 0.50, (0, 4, 0, -6),
            "Index and thumb circle, others extended"),
    Gesture("I", "Letter I", (0.95, 0.95, 0.9, 0.05), 110, 0.50, (0, 0, 0, -8),
            "Pinky up, fist closed"),
    Gesture("L", "Letter L", (0.03, 0.95, 0.95, 0.95), 5, 0.02,
            meaning="Index up, thumb out — L shape"),
    Gesture("O", "Letter O", (0.55, 0.60, 0.60, 0.60), 25, 0.55, (2, 0, -2, -4),
            "Fingertips meet thumb in an O"),
    Gesture("V", "Letter V", (0.03, 0.03, 0.95, 0.95), 115, 0.50, (14, -6, 0, -4),
            "Index and middle apart — victory sign"),
    Gesture("W", "Letter W", (0.03, 0.03, 0.03, 0.9), 120, 0.60, (12, 0, -12, -6),
            "Index, middle, ring extended"),
    Gesture("Y", "Letter Y", (0.95, 0.95, 0.95, 0.03), 5, 0.02, (0, 0, 0, -14),
            "Thumb and pinky out — hang loose"),
    Gesture("HELLO", "Hello 👋", (0.02, 0.02, 0.02, 0.02), 25, 0.05, (8, 2, -4, -10),
            "Open palm, all fingers spread"),
    Gesture("YES", "Yes / Good 👍", (1, 1, 1, 1), 85, 0.0,
            meaning="Fist with thumb extended up"),
    Gesture("I_LOVE_YOU", "I Love You 🤟", (0.03, 0.95, 0.95, 0.03), 10, 0.02,
            (8, 0, 0, -12), "Thumb, index and pinky extended"),
]
GESTURE_BY_NAME = {g.name: g for g in GESTURES}
CLASS_NAMES = [g.name for g in GESTURES]


def _chain(base, lengths, start_deg, bend_degs):
    """Build a kinematic chain of joints; each segment bends further toward the palm."""
    pts = [np.array(base, dtype=float)]
    ang = math.radians(start_deg)
    for length, bend in zip(lengths, bend_degs):
        ang -= math.radians(bend)
        pts.append(pts[-1] + length * np.array([math.cos(ang), math.sin(ang)]))
    return pts


def synth_landmarks(g: Gesture, rng=None, jitter=0.0):
    """Generate (21, 2) landmark array for a gesture, with optional parameter jitter."""
    rng = rng or np.random.default_rng()
    j = lambda s: rng.normal(0, s) if jitter else 0.0
    pts = [np.zeros(2)]
    t_ang = g.thumb_ang + j(6 * jitter)
    t_curl = min(1.0, max(0.0, g.thumb_curl + j(0.08 * jitter)))
    base = np.array(THUMB_BASE) + np.array([j(0.02), j(0.02)])
    pts += _chain(base, THUMB_LENGTHS, t_ang, [t_curl * 55, t_curl * 65, t_curl * 45])
    for i, name in enumerate(("index", "middle", "ring", "pinky")):
        curl = min(1.0, max(0.0, g.curls[i] + j(0.07 * jitter)))
        splay = g.splays[i] + j(3 * jitter)
        mcp = np.array(FINGER_MCPS[name]) + np.array([j(0.02), j(0.02)])
        chain = _chain(mcp, FINGER_LENGTHS[name], 90 + splay,
                       [curl * 95, curl * 105, curl * 70])
        pts += chain
    arr = np.array(pts)
    if jitter:
        arr = arr + rng.normal(0, 0.012 * jitter, arr.shape)  # coordinate noise
    return arr

"""Smoke test: baked model + full inference path. Run: python smoke_test.py"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.hand_model import GESTURES, synth_landmarks
from src.predictor import SignPredictor, landmarks_from_image

p = SignPredictor()
ok = 0
for f in sorted(Path("samples").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    preds = p.predict(np.array(d["landmarks"]))
    correct = preds[0][0] == d["gesture"]
    ok += correct
    print(f"{d['gesture']:12s} -> {preds[0][0]:12s} p={preds[0][2]:.2f} "
          f"{'OK' if correct else 'WRONG'}")
print(f"Canonical samples: {ok}/14 correct")
assert ok == 14, "canonical sample misclassified"

rng = np.random.default_rng(7)
hits = total = 0
for g in GESTURES:
    for _ in range(30):
        lms = synth_landmarks(g, rng=rng, jitter=1.0)
        if rng.random() < 0.5:
            lms = lms * np.array([-1.0, 1.0])
        hits += p.predict(lms)[0][0] == g.name
        total += 1
print(f"Jittered robustness: {hits}/{total} = {hits / total:.1%}")
assert hits / total > 0.9

img = Image.fromarray(
    np.random.default_rng(0).uniform(0, 255, (480, 640, 3)).astype("uint8"))
pts, hand = landmarks_from_image(img)
print("MediaPipe no-hand path returns None:", pts is None and hand is None)
print("ALL SMOKE TESTS PASSED")

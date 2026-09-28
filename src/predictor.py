"""Model loading and prediction. MediaPipe is imported lazily so the demo
gallery and training work even where MediaPipe is unavailable."""
import json
from pathlib import Path

import joblib
import numpy as np

from src.features import extract_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class SignPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        # Safe: loads only the project's own baked model artifact (trained and
        # shipped inside this repo/image by train_model.py), never user input.
        self.model = joblib.load(models_dir / "sign_model.pkl")
        meta = json.loads((models_dir / "label_map.json").read_text(encoding="utf-8"))
        # predict_proba columns follow model.classes_ (sklearn sorts labels),
        # NOT the template-definition order in label_map.json.
        self.classes = list(self.model.classes_)
        self.display = meta["display"]

    def predict(self, landmarks, top_k=3):
        """Return [(class_name, display_name, probability), ...] best-first."""
        feats = extract_features(landmarks).reshape(1, -1)
        probs = self.model.predict_proba(feats)[0]
        order = np.argsort(probs)[::-1][:top_k]
        return [(self.classes[i], self.display[self.classes[i]], float(probs[i]))
                for i in order]


def landmarks_from_image(pil_image):
    """Detect one hand in a PIL image via MediaPipe. Returns ((21,2) pixel-space
    landmarks, handedness_label) or (None, None) when no hand is found."""
    import mediapipe as mp  # lazy: heavy import
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                  min_detection_confidence=0.4) as hands:
        res = hands.process(img)
    if not res.multi_hand_landmarks:
        return None, None
    lm = res.multi_hand_landmarks[0].landmark
    pts = np.array([[p.x * w, (1 - p.y) * h] for p in lm])  # flip y: image->math coords
    label = res.multi_handedness[0].classification[0].label if res.multi_handedness else "?"
    return pts, label

"""Train the sign classifier on a procedurally synthesized landmark dataset.

Generates augmented samples per gesture (parameter jitter + coordinate noise +
handedness mirroring handled by normalization), trains a RandomForest, evaluates
on a holdout split, and bakes: models/sign_model.pkl, label_map.json,
metrics.json, confusion_matrix.png, plus canonical demo samples in samples/.

Run: python train_model.py            (~30s CPU, fully offline)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features import extract_features
from src.hand_model import CLASS_NAMES, GESTURES, synth_landmarks
from src.visualize import render_skeleton

ROOT = Path(__file__).resolve().parent
SAMPLES_PER_CLASS = 500
SEED = 42


def build_dataset(rng):
    X, y = [], []
    for g in GESTURES:
        for _ in range(SAMPLES_PER_CLASS):
            lms = synth_landmarks(g, rng=rng, jitter=1.0)
            if rng.random() < 0.5:  # random handedness; normalize() re-mirrors
                lms = lms * np.array([-1.0, 1.0])
            X.append(extract_features(lms))
            y.append(g.name)
    return np.array(X), np.array(y)


def plot_confusion(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted"), ax.set_ylabel("True")
    ax.set_title("Sign Classifier — Confusion Matrix (holdout)")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    print(f"Generating dataset: {len(GESTURES)} classes x {SAMPLES_PER_CLASS} samples...")
    X, y = build_dataset(rng)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    model = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    print("Training RandomForest (300 trees)...")
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    print(f"Holdout accuracy: {acc:.4f} | 5-fold CV: {cv.mean():.4f} +/- {cv.std():.4f}")
    report = classification_report(y_te, y_pred, output_dict=True)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "sign_model.pkl", compress=3)
    (models_dir / "label_map.json").write_text(json.dumps({
        "classes": CLASS_NAMES,
        "display": {g.name: g.display for g in GESTURES},
        "meaning": {g.name: g.meaning for g in GESTURES},
    }, indent=2), encoding="utf-8")
    (models_dir / "metrics.json").write_text(json.dumps({
        "model": "RandomForestClassifier(n_estimators=300)",
        "features": int(X.shape[1]), "samples": int(X.shape[0]),
        "classes": len(CLASS_NAMES), "holdout_accuracy": round(acc, 4),
        "cv_mean": round(float(cv.mean()), 4), "cv_std": round(float(cv.std()), 4),
        "per_class_f1": {k: round(v["f1-score"], 4) for k, v in report.items()
                         if k in CLASS_NAMES},
        "note": "Trained on procedurally synthesized + augmented landmark data "
                "(demonstrative). Retrain on real captures for production use.",
    }, indent=2), encoding="utf-8")
    plot_confusion(confusion_matrix(y_te, y_pred, labels=CLASS_NAMES),
                   CLASS_NAMES, models_dir / "confusion_matrix.png")

    samples_dir = ROOT / "samples"
    samples_dir.mkdir(exist_ok=True)
    for g in GESTURES:  # canonical (unjittered) demo poses
        lms = synth_landmarks(g)
        (samples_dir / f"{g.name}.json").write_text(
            json.dumps({"gesture": g.name, "landmarks": lms.tolist()}), encoding="utf-8")
        render_skeleton(lms).save(samples_dir / f"{g.name}.png")
    print(f"Saved model + metrics to {models_dir}, demo samples to {samples_dir}")


if __name__ == "__main__":
    main()

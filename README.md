# AI-Driven Sign Language Translator

**M.Tech Final Year Project — Roll No. 23B21DAH13 (KIET)**

Translates static sign-language hand gestures from images into text using
hand-landmark geometry and machine learning, with a sentence builder that
composes recognized signs into readable output.

## Abstract

Communication barriers between the deaf/hard-of-hearing community and
non-signers motivate automated sign-language translation. This project
implements a real-time, camera-optional sign recognition pipeline: **MediaPipe
Hands** extracts 21 3-D hand landmarks per frame; a geometric normalization
stage makes the representation invariant to translation, rotation, scale and
handedness; a 56-dimensional feature vector (normalized joint coordinates,
per-finger curl distances, inter-fingertip gaps, thumb orientation) feeds a
**Random Forest** classifier that recognizes **14 static signs** (ASL letters
A B C D F I L O V W Y plus *Hello*, *Yes/Good*, *I Love You*). Recognized signs
stream into a sentence builder, producing text translation. The system runs
fully offline in a single CPU-only Docker container with a Streamlit interface.

## System architecture

```
                 ┌────────────────────────────────────────────────┐
 Input           │ Feature pipeline                               │        Output
 ─────           │                                                │        ──────
 Upload ─┐       │  MediaPipe Hands      Normalization            │
 Camera ─┼──────▶│  21 landmarks   ────▶ (translate/rotate/scale/ │
 Demo   ─┘       │  (x, y per joint)      mirror-invariant)       │
 gallery         │            │                                   │
                 │            ▼                                   │
                 │  56-dim features ───▶ RandomForest (300 trees) │──▶ sign label
                 │  coords + curls +      per-class probabilities │        │
                 │  gaps + thumb vec                              │        ▼
                 └────────────────────────────────────────────────┘  sentence builder
                                                                      → translated text
```

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI: demo gallery, upload, browser camera, model report |
| `src/hand_model.py` | Parametric 21-joint hand model + 14 gesture templates |
| `src/features.py` | Landmark normalization + feature extraction (train & infer) |
| `src/predictor.py` | Model loading, prediction, MediaPipe image inference |
| `src/visualize.py` | Skeleton rendering / photo overlay |
| `train_model.py` | Dataset synthesis, training, evaluation, artifact baking |
| `models/` | Baked weights `sign_model.pkl`, `metrics.json`, confusion matrix |
| `samples/` | Canonical landmark JSON + PNG per sign (self-running demo) |

## Methodology

1. **Landmark acquisition** — MediaPipe Hands (static image mode, 1 hand),
   pixel-space landmarks; y-axis flipped to math coordinates.
2. **Normalization** — wrist translated to origin; wrist→middle-MCP vector
   rotated to +y; scaled to unit palm length; left hands mirrored to right.
   This removes camera pose and handedness as nuisance factors.
3. **Features (56)** — 42 normalized coordinates; 5 fingertip–wrist distances;
   4 fingertip–MCP distances (finger curl); 3 adjacent fingertip gaps
   (e.g. separates V from W); thumb TIP–MCP vector (separates A/Yes/L/Y).
4. **Training data** — each gesture is defined as joint-angle parameters
   (per-finger curl, splay, thumb angle); 500 samples/class are synthesized
   with joint-angle jitter, coordinate noise and random handedness (7,000
   samples total). This makes the pipeline dataset-agnostic: swapping in real
   captured landmarks and re-running `train_model.py` retrains identically.
5. **Model** — Random Forest, 300 trees; evaluated on a stratified 20% holdout
   plus 5-fold cross-validation (see `models/metrics.json` and the in-app
   *Model & Training* tab for the exact numbers and confusion matrix).

## Results

See `models/metrics.json` (regenerated on every training run) — holdout
accuracy, 5-fold CV mean ± std, and per-class F1 for all 14 signs, plus
`models/confusion_matrix.png`.

## Limitations & future work

- Static signs only; dynamic gestures (J, Z, full words) need temporal models
  (LSTM/Transformer over landmark sequences).
- The classifier is trained on synthesized landmark geometry — demonstrative
  by design; collecting a real capture set and re-running `train_model.py`
  is the direct path to production accuracy.
- Single-hand vocabulary; two-handed signs require multi-hand fusion.
- Text-to-speech and Indian Sign Language (ISL) vocabulary are natural
  extensions.

## Quick start

See `HOW_TO_RUN.md` — one command via Docker (`run.ps1` / `run.sh`), opens at
http://localhost:8501. No internet, GPU, or webcam required for the demo.

"""AI-Driven Sign Language Translator — Streamlit UI (port 8501).

Tabs: self-running Demo Gallery (no camera needed), Upload Image, browser
Camera, Model & Training report, About. A sidebar sentence builder turns
recognized signs into text ("translation").
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.hand_model import GESTURE_BY_NAME
from src.predictor import SignPredictor, landmarks_from_image
from src.visualize import overlay_skeleton, render_skeleton

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AI Sign Language Translator", page_icon="🤟",
                   layout="wide")


@st.cache_resource
def load_predictor():
    return SignPredictor()


def show_prediction(preds, key):
    best_name, best_display, best_p = preds[0]
    meaning = GESTURE_BY_NAME[best_name].meaning
    st.success(f"**Detected sign: {best_display}**  (confidence {best_p:.1%})")
    if meaning:
        st.caption(meaning)
    st.bar_chart({d: p for _, d, p in preds}, horizontal=True)
    word = best_display.split()[0] if best_name in ("HELLO", "YES") else (
        "I love you" if best_name == "I_LOVE_YOU" else best_name)
    if st.button(f"➕ Add “{word}” to sentence", key=key):
        st.session_state.sentence.append(word)
        st.rerun()


def classify_image(img, predictor, key):
    with st.spinner("Detecting hand landmarks (MediaPipe)..."):
        pts, handedness = landmarks_from_image(img)
    if pts is None:
        st.warning("No hand detected. Use a clear, well-lit photo with one hand "
                   "filling most of the frame, palm facing the camera.")
        return
    c1, c2 = st.columns(2)
    c1.image(overlay_skeleton(img, pts), caption=f"Detected hand ({handedness})",
             use_container_width=True)
    with c2:
        show_prediction(predictor.predict(pts), key)


def sidebar(predictor):
    st.sidebar.title("🤟 Sign Translator")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    st.sidebar.metric("Model accuracy (holdout)", f"{metrics['holdout_accuracy']:.1%}")
    st.sidebar.caption(f"{metrics['classes']} signs · {metrics['samples']} training "
                       f"samples · {metrics['features']} features")
    st.sidebar.divider()
    st.sidebar.subheader("📝 Sentence builder")
    sentence = " ".join(st.session_state.sentence) or "—"
    st.sidebar.markdown(f"### {sentence}")
    if st.sidebar.button("🗑️ Clear sentence"):
        st.session_state.sentence = []
        st.rerun()


def demo_tab(predictor):
    st.subheader("Self-running demo — no camera required")
    names = predictor.classes
    display = predictor.display
    choice = st.selectbox("Choose a sign to demo",
                          names, format_func=lambda n: display[n])
    data = json.loads((ROOT / "samples" / f"{choice}.json").read_text(encoding="utf-8"))
    pts = np.array(data["landmarks"])
    c1, c2 = st.columns(2)
    c1.image(render_skeleton(pts), caption=f"Canonical hand pose: {display[choice]}",
             use_container_width=True)
    with c2:
        show_prediction(predictor.predict(pts), key=f"demo-{choice}")


def model_tab():
    st.subheader("Model & training report")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holdout accuracy", f"{metrics['holdout_accuracy']:.2%}")
    c2.metric("5-fold CV", f"{metrics['cv_mean']:.2%} ± {metrics['cv_std']:.2%}")
    c3.metric("Classes", metrics["classes"])
    c4.metric("Training samples", metrics["samples"])
    st.image(str(ROOT / "models" / "confusion_matrix.png"),
             caption="Confusion matrix on the 20% holdout split")
    st.info(metrics["note"])
    st.json(metrics["per_class_f1"], expanded=False)


def about_tab():
    st.subheader("About this project")
    st.markdown("""
**AI-Driven Sign Language Translator** — M.Tech final-year project (23B21DAH13, KIET).

**Pipeline:** Image → **MediaPipe Hands** (21 landmarks) → geometric normalization
(translation/rotation/scale/handedness invariant) → 56-dim feature vector
(coordinates + finger-curl distances + inter-fingertip gaps + thumb vector) →
**RandomForest** classifier → sign label → sentence builder (translation to text).

**Recognized vocabulary:** ASL letters A B C D F I L O V W Y and the signs
Hello 👋, Yes/Good 👍, I Love You 🤟 (14 static signs).

**Training data:** procedurally synthesized from a parametric 21-joint hand model
with heavy augmentation (joint-angle jitter, coordinate noise, handedness
mirroring). The pipeline is dataset-agnostic — re-run `train_model.py` after
swapping in real captured landmarks to productionize.
""")


def main():
    st.session_state.setdefault("sentence", [])
    predictor = load_predictor()
    sidebar(predictor)
    st.title("AI-Driven Sign Language Translator")
    tabs = st.tabs(["🖐️ Demo Gallery", "📤 Upload Image", "📷 Camera",
                    "📊 Model & Training", "ℹ️ About"])
    with tabs[0]:
        demo_tab(predictor)
    with tabs[1]:
        up = st.file_uploader("Upload a photo of a hand sign",
                              type=["jpg", "jpeg", "png", "webp"])
        if up:
            classify_image(Image.open(up), predictor, key="upload")
    with tabs[2]:
        st.caption("Uses your browser's camera — works even when the app runs "
                   "inside Docker.")
        shot = st.camera_input("Show a sign to the camera")
        if shot:
            classify_image(Image.open(shot), predictor, key="camera")
    with tabs[3]:
        model_tab()
    with tabs[4]:
        about_tab()


if __name__ == "__main__":
    main()

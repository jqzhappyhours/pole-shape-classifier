from __future__ import annotations

import os
import tempfile

import streamlit as st

from pole_infer import VideoInferenceConfig, load_model, predict_video


st.set_page_config(page_title="Pole shape classifier", page_icon="🎥", layout="centered")

st.title("Pole combo video → pole shape")
st.caption("Upload a video and the model will classify: inside vs outside leg hang.")


@st.cache_resource
def _get_model(model_path: str | None):
    return load_model(model_path)


with st.sidebar:
    st.header("Settings")
    model_path = st.text_input(
        "Model path (optional)",
        value=os.environ.get("POLE_MODEL_PATH", ""),
        help="Leave blank to auto-load `efficientnetb0_v2.keras` or `efficientnetb0_v1.keras` from repo root.",
    ).strip() or None

    frame_interval = st.slider("Frame interval (every Nth frame)", 1, 120, 30)
    max_frames = st.slider("Max frames to analyze", 4, 128, 64)

    config = VideoInferenceConfig(frame_interval=frame_interval, max_frames=max_frames)


uploaded = st.file_uploader(
    "Upload a pole combo video",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload a video to get a prediction.")
    st.stop()

suffix = f".{uploaded.name.split('.')[-1]}" if "." in uploaded.name else ".mp4"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
    f.write(uploaded.getbuffer())
    tmp_path = f.name

try:
    with st.spinner("Loading model…"):
        model = _get_model(model_path)

    with st.spinner("Running inference…"):
        result = predict_video(model, tmp_path, config=config)

    st.subheader("Prediction")
    st.metric("Pole shape", result["pred_name"])
    st.write(f"Confidence: **{result['confidence']:.3f}**")
    st.caption(
        f"Analyzed {result['n_frames']} frames • interval={result['frame_interval']} • max_frames={result['max_frames']}"
    )

    with st.expander("Debug (mean probabilities)"):
        st.json(result)
finally:
    try:
        os.remove(tmp_path)
    except OSError:
        pass


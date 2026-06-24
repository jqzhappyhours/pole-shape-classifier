from __future__ import annotations

import os
import tempfile

import streamlit as st

from pole_infer import (
    VideoInferenceConfig,
    annotate_video,
    load_mlp_model,
    predict_video_sequence_coords,
)


st.set_page_config(page_title="Pole shape classifier", page_icon="🎥", layout="centered")

st.title("Pole combo video → pole shape")
st.caption("Upload a video and the model will classify pole shapes using pose landmarks.")


@st.cache_resource
def _get_model(model_path: str | None):
    return load_mlp_model(model_path)


with st.sidebar:
    st.header("Settings")
    model_path = st.text_input(
        "Model path (optional)",
        value=os.environ.get("POLE_MODEL_PATH", ""),
        help="Leave blank to auto-load `pose_mlp.keras` from repo root.",
    ).strip() or None

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Predictions below this confidence are labelled 'unknown'.",
    )

    frame_interval = st.slider(
        "Prediction interval (frames)",
        min_value=10,
        max_value=90,
        value=30,
        step=10,
        help="How many frames to skip between predictions. 30 ≈ 1s at 30fps, 60 ≈ 2s.",
    )

config = VideoInferenceConfig(
    frame_interval=frame_interval,
    max_frames=120,
    smooth_window=5,
    confidence_threshold=confidence_threshold,
)


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

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
    annotated_path = f.name

try:
    with st.spinner("Loading model…"):
        model = _get_model(model_path)

    with st.spinner("Running inference…"):
        segments = predict_video_sequence_coords(model, tmp_path, config=config)

    with st.spinner("Annotating video…"):
        annotate_video(tmp_path, segments, annotated_path)

    st.subheader("Annotated video")
    with open(annotated_path, "rb") as f:
        st.video(f.read())

    st.subheader("Shape sequence")
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        label = seg["label"]
        conf = seg["confidence"]
        duration = end - start
        st.markdown(
            f"**{label}** &nbsp; `{start:.1f}s – {end:.1f}s` &nbsp; ({duration:.1f}s) &nbsp; conf: {conf:.2f}"
        )
finally:
    for path in (tmp_path, annotated_path):
        try:
            os.remove(path)
        except OSError:
            pass

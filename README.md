# Pole shape classifier

A Streamlit app that classifies pole dance shapes from video using a pose landmark MLP model.

**Supported shapes:** `airwalk`, `climb`, `inside_leg_hang`, `invert`, `outside_leg_hang`, `pencil`

## How it works

1. Each video frame is sampled at a configurable interval
2. [MediaPipe](https://developers.google.com/mediapipe) detects 33 body landmarks per frame
3. Landmarks are normalised (centred on hip midpoint, scaled by torso length) into a 99-dim vector
4. A small MLP classifies the vector into one of 6 pole shapes
5. Consecutive identical predictions are merged into labelled time segments
6. Predictions below the confidence threshold are labelled `unknown`

## Setup

Use **Python 3.12** (3.10–3.12). TensorFlow has no wheels for 3.14+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.12` is not found: `brew install python@3.12`

## Train the model

Open `model.ipynb` and run all cells in the **Landmark-based MLP** section.

Before running the notebook, extract pose coordinates from your training videos:

```bash
python extract.py --coords
```

This writes `data/coords/train.csv`, `val.csv`, and `test.csv`. The trained model is saved as `pose_mlp.keras`.

To add more training data for a specific shape:

```bash
# Add videos to data/clips/<shape>/ then re-extract
python extract.py --coords
```

## Run the app

```bash
streamlit run app.py
```

Place `pose_mlp.keras` in the repo root (same folder as `app.py`), or set `POLE_MODEL_PATH` / enter a path in the sidebar.

## Sidebar controls

| Setting | Default | Description |
|---|---|---|
| Model path | auto | Path to `pose_mlp.keras`, or leave blank to auto-load |
| Confidence threshold | 0.6 | Frames below this confidence are labelled `unknown` |
| Prediction interval | 30 frames | Frames skipped between predictions (~1s at 30fps) |

## Project structure

```
app.py              — Streamlit app
pole_infer.py       — Inference functions (load model, predict from video)
pose_utils.py       — MediaPipe landmark extraction
extract.py          — Extract frames / coords from training videos
model.ipynb         — Model training notebook
data/
  clips/<shape>/    — Raw training videos per shape
  coords/           — Extracted landmark CSVs (train/val/test)
  images/           — Extracted image frames (for EfficientNet baseline)
```

# Pole shape classifier (video upload app)

This repo contains a trained image classifier (frames) for:

- `inside_leg_hang`
- `outside_leg_hang`

and a small app that lets a user upload a **video**, samples frames, and returns one predicted label.

## 1) Put your trained model in the repo root

Your notebook saves models like:

- `efficientnetb0_v1.keras`
- `efficientnetb0_v2.keras`

Place one of those files in the repo root (same folder as `app.py`).

Alternatively, set `POLE_MODEL_PATH` or type a path in the app sidebar.

## 2) Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- TensorFlow has strict Python version support. If `pip install tensorflow` fails, create the venv with a supported Python (commonly `python3.10`, `python3.11`, or `python3.12`, depending on your platform/TensorFlow version).
- Frame preprocessing matches your training setup: resize to **224×224** and keep pixel values in **[0, 255]** (no extra normalization).
- Video inference averages probabilities across sampled frames.


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

Use **Python 3.12** (or 3.10–3.12). Do **not** use Python 3.14 — TensorFlow has no wheels for it yet.

On macOS with Homebrew:

```bash
# Remove a broken venv if you previously ran `python3 -m venv` with Python 3.14
rm -rf .venv

python3.12 -m venv .venv
source .venv/bin/activate

# Always install with the venv's python (avoids pip/python version mismatch)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

streamlit run app.py
```

If `python3.12` is not found: `brew install python@3.12`

## Notes

- If you see `Could not find a version that satisfies tensorflow`, your venv was likely created with Python 3.14. Recreate it with `python3.12` as above.
- After activating the venv, run `python --version` — it should show 3.12.x, not 3.14.
- Frame preprocessing matches your training setup: resize to **224×224** and keep pixel values in **[0, 255]** (no extra normalization).
- Video inference averages probabilities across sampled frames.


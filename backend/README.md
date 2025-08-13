# Responsive CSS AI Backend

A FastAPI backend that predicts a responsive CSS scale and returns CSS variables/snippets based on device/display inputs using a small neural network (scikit-learn MLP).

## Quickstart

1. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

2. Train the model (synthetic data):

```bash
python -m backend.model.train
```

3. Run the API server:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- GET `/health` → `{ status, model_present }`
- POST `/predict` with JSON body:

```json
{
  "width_px": 3840,
  "height_px": 2160,
  "diagonal_in": 27,
  "refresh_hz": 144
}
```

Response includes a scale, tokens map, and a ready-to-use CSS string.

- GET `/predict?width_px=3840&height_px=2160&diagonal_in=27&refresh_hz=144` same as POST.

## Output

- `scale`: scalar in [0.8, 1.8]
- `tokens`: CSS custom properties (e.g., `--font-size-base`, `--space-*`)
- `css`: `:root { ... }` and a default `html { font-size: var(--font-size-base); }`

## Notes

- Model is trained on synthetic data derived from ergonomic heuristics; tune `backend/model/data.py` for different behaviors.
- If the model artifact is missing on startup, the app attempts to train one automatically.
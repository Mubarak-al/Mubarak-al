from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

# Optional deps loaded lazily
try:
    import numpy as np  # type: ignore
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
    joblib = None  # type: ignore

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _compute_features(width_px: int, height_px: int, diagonal_in: float, refresh_hz: int):
    aspect_ratio = width_px / height_px
    ppi = math.sqrt(width_px**2 + height_px**2) / max(1e-6, diagonal_in)
    megapixels = (width_px * height_px) / 1_000_000.0

    return [
        float(width_px),
        float(height_px),
        float(diagonal_in),
        float(refresh_hz),
        float(aspect_ratio),
        float(ppi),
        float(megapixels),
    ]


def _predict_scale_heuristic(width_px: int, height_px: int, diagonal_in: float, refresh_hz: int) -> float:
    w = float(width_px)
    h = float(height_px)
    diag = float(diagonal_in)
    hz = float(refresh_hz)

    ppi = math.sqrt(w*w + h*h) / max(1e-6, diag)

    width_factor = _sigmoid((w - 1366.0) / 500.0)
    density_factor = _sigmoid((ppi - 150.0) / 50.0)
    small_device_penalty = -0.25 * _sigmoid((8.0 - diag) / 1.5)
    large_display_bonus = 0.20 * _sigmoid((diag - 24.0) / 5.0)
    refresh_bonus = 0.10 * _sigmoid((hz - 60.0) / 80.0) - 0.05

    s = 1.0 + 0.35 * width_factor + 0.25 * density_factor + small_device_penalty + large_display_bonus + refresh_bonus
    return max(0.8, min(1.8, s))


def predict_scale(width_px: int, height_px: int, diagonal_in: float, refresh_hz: int) -> float:
    # Try model if available and deps present; otherwise fallback heuristic
    if MODEL_PATH.exists() and joblib is not None and np is not None:
        try:
            model = joblib.load(MODEL_PATH)
            import numpy as _np  # local alias
            X = _np.array([_compute_features(width_px, height_px, diagonal_in, refresh_hz)], dtype=_np.float32)
            scale = float(model.predict(X)[0])
            return max(0.8, min(1.8, scale))
        except Exception:
            pass
    return _predict_scale_heuristic(width_px, height_px, diagonal_in, refresh_hz)


def _round(value: float) -> float:
    return float(round(value, 2))


def build_tokens_from_scale(scale: float) -> Dict[str, str]:
    base_font_px = max(12.0, min(24.0, 16.0 * scale))
    base_unit_px = max(2.0, 4.0 * scale)

    spacing = {
        "--space-1": _round(1 * base_unit_px),
        "--space-2": _round(2 * base_unit_px),
        "--space-3": _round(3 * base_unit_px),
        "--space-4": _round(4 * base_unit_px),
        "--space-5": _round(6 * base_unit_px),
        "--space-6": _round(8 * base_unit_px),
        "--space-7": _round(12 * base_unit_px),
        "--space-8": _round(16 * base_unit_px),
    }

    ratio = 1.125
    type_scale = {
        "--font-size-xs": _round(base_font_px / (ratio**2)),
        "--font-size-sm": _round(base_font_px / ratio),
        "--font-size-md": _round(base_font_px),
        "--font-size-lg": _round(base_font_px * ratio),
        "--font-size-xl": _round(base_font_px * (ratio**2)),
        "--font-size-2xl": _round(base_font_px * (ratio**3)),
        "--font-size-3xl": _round(base_font_px * (ratio**4)),
    }

    misc = {
        "--radius-sm": _round(2.0 * scale),
        "--radius-md": _round(4.0 * scale),
        "--radius-lg": _round(8.0 * scale),
        "--elevation-1": _round(2.0 * scale),
        "--elevation-2": _round(4.0 * scale),
        "--elevation-3": _round(8.0 * scale),
    }

    tokens = {
        "--scale": str(_round(scale)),
        "--font-size-base": f"{_round(base_font_px)}px",
        **{k: f"{v}px" for k, v in spacing.items()},
        **{k: f"{v}px" for k, v in type_scale.items()},
        **{k: f"{v}px" for k, v in misc.items()},
    }

    return tokens


def tokens_to_css(tokens: Dict[str, str]) -> str:
    lines = [":root {"]
    for key, value in tokens.items():
        lines.append(f"  {key}: {value};")
    lines.append("}")
    lines.append("")
    lines.append("html { font-size: var(--font-size-base); }")
    return "\n".join(lines)


essential_params = ("width_px", "height_px", "diagonal_in", "refresh_hz")


def predict_css(
    width_px: int,
    height_px: int,
    diagonal_in: float,
    refresh_hz: int,
) -> Tuple[float, Dict[str, str], str]:
    scale = predict_scale(width_px, height_px, diagonal_in, refresh_hz)
    tokens = build_tokens_from_scale(scale)
    css = tokens_to_css(tokens)
    return scale, tokens, css
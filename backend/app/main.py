from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from backend.model.inference import predict_css, MODEL_PATH

app = FastAPI(title="Responsive CSS AI", version="0.1.0")


class PredictRequest(BaseModel):
    width_px: int = Field(..., ge=200, le=10000)
    height_px: int = Field(..., ge=200, le=10000)
    diagonal_in: float = Field(..., gt=1, lt=100)
    refresh_hz: int = Field(..., ge=30, le=1000)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_present": Path(MODEL_PATH).exists(),
    }


@app.post("/predict")
async def predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        scale, tokens, css = predict_css(
            req.width_px, req.height_px, req.diagonal_in, req.refresh_hz
        )
        return {
            "scale": scale,
            "tokens": tokens,
            "css": css,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict")
async def predict_query(
    width_px: int = Query(..., ge=200, le=10000),
    height_px: int = Query(..., ge=200, le=10000),
    diagonal_in: float = Query(..., gt=1, lt=100),
    refresh_hz: int = Query(60, ge=30, le=1000),
) -> Dict[str, Any]:
    try:
        scale, tokens, css = predict_css(width_px, height_px, diagonal_in, refresh_hz)
        return {
            "scale": scale,
            "tokens": tokens,
            "css": css,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
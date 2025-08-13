from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from .data import generate_synthetic_dataset


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "model.joblib"


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_and_save_model(num_samples: int = 10000, random_seed: int = 42) -> Tuple[Pipeline, dict]:
    X, y, _ = generate_synthetic_dataset(num_samples=num_samples, random_seed=random_seed)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )

    pipeline = build_pipeline(random_state=random_seed)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    metrics = {
        "r2": float(r2_score(y_val, y_pred)),
        "mae": float(mean_absolute_error(y_val, y_pred)),
    }

    joblib.dump(pipeline, MODEL_PATH)

    return pipeline, metrics


if __name__ == "__main__":
    pipeline, metrics = train_and_save_model()
    print("Saved model to:", MODEL_PATH)
    print("Validation metrics:", metrics)
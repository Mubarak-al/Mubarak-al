import math
import random
from typing import List, Tuple

import numpy as np


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_synthetic_dataset(
    num_samples: int = 8000,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Common aspect ratios
    aspect_ratios = [16/9, 16/10, 3/2, 4/3, 20/9, 21/9]

    widths = np.random.uniform(320, 3840, size=num_samples)
    ratios = np.random.choice(aspect_ratios, size=num_samples)
    heights = widths / ratios

    # Device diagonal sizes by category (in inches)
    diag_choices = [
        (5.0, 7.0),   # phones
        (8.0, 13.0),  # tablets
        (13.0, 18.0), # laptops
        (22.0, 38.0), # desktop monitors
    ]
    diagonals = np.array([
        np.random.uniform(*random.choice(diag_choices)) for _ in range(num_samples)
    ])

    refresh_pool = [60, 75, 90, 120, 144, 165, 240, 267]
    refresh_probs = np.array([0.20, 0.10, 0.15, 0.20, 0.15, 0.10, 0.08, 0.02])
    refresh_probs = refresh_probs / refresh_probs.sum()
    refresh_rates = np.random.choice(refresh_pool, size=num_samples, p=refresh_probs)

    # Derived features
    ppi = np.sqrt(widths**2 + heights**2) / diagonals
    aspect = widths / heights
    megapixels = (widths * heights) / 1_000_000.0

    # Synthetic target: scale factor s in [0.8, 1.8]
    width_factor = _sigmoid((widths - 1366.0) / 500.0)  # grows with width
    density_factor = _sigmoid((ppi - 150.0) / 50.0)     # grows with ppi
    small_device_penalty = -0.25 * _sigmoid((8.0 - diagonals) / 1.5)
    large_display_bonus = 0.20 * _sigmoid((diagonals - 24.0) / 5.0)
    refresh_bonus = 0.10 * _sigmoid((refresh_rates - 60.0) / 80.0) - 0.05

    s = 1.0 + 0.35 * width_factor + 0.25 * density_factor + small_device_penalty + large_display_bonus + refresh_bonus
    s = np.clip(s, 0.8, 1.8)

    # Feature matrix
    X = np.stack([
        widths, heights, diagonals, refresh_rates, aspect, ppi, megapixels
    ], axis=1)
    y = s

    feature_names = [
        "width_px",
        "height_px",
        "diagonal_in",
        "refresh_hz",
        "aspect_ratio",
        "ppi",
        "megapixels",
    ]

    return X.astype(np.float32), y.astype(np.float32), feature_names
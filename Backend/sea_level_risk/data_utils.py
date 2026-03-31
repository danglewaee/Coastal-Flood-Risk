import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_series(csv_path: str, time_col: str | None = None, value_col: str = "sea_level") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if value_col not in df.columns:
        raise ValueError(f"Missing required column: {value_col}")

    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col)

    df = df[[c for c in [time_col, value_col] if c is not None and c in df.columns]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).reset_index(drop=True)
    return df


def zscore_normalize(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0.0:
        std = 1.0
    normalized = (values - mean) / std
    return normalized, mean, std


def apply_zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    safe_std = std if std != 0.0 else 1.0
    return (values - mean) / safe_std


def invert_zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def create_supervised_sequences(
    series: np.ndarray,
    lookback: int,
    targets: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(series, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    if features.shape[0] <= lookback:
        raise ValueError(f"Not enough rows: need > {lookback}, got {features.shape[0]}")

    target_arr = np.asarray(targets, dtype=np.float32).reshape(-1) if targets is not None else features[:, 0]
    if target_arr.shape[0] != features.shape[0]:
        raise ValueError("Feature rows and target rows must match.")

    x_data, y_data = [], []
    for i in range(features.shape[0] - lookback):
        x_data.append(features[i : i + lookback])
        y_data.append(target_arr[i + lookback])

    x_arr = np.array(x_data, dtype=np.float32)
    y_arr = np.array(y_data, dtype=np.float32).reshape(-1, 1)
    return x_arr, y_arr


def save_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .data_utils import apply_zscore, invert_zscore, load_metadata
from .features import build_feature_frame, resolve_feature_mode
from .uncertainty import calibrated_quantile_forecast


def _coerce_recent_timestamps(recent_timestamps: list | np.ndarray | pd.Series | None, recent_values: np.ndarray) -> pd.DatetimeIndex:
    if recent_timestamps is None:
        return pd.date_range("2000-01-01", periods=recent_values.shape[0], freq="1h", tz="UTC")

    ts = pd.to_datetime(pd.Series(recent_timestamps), errors="coerce", utc=True)
    if ts.shape[0] != recent_values.shape[0] or ts.isna().any():
        return pd.date_range("2000-01-01", periods=recent_values.shape[0], freq="1h", tz="UTC")
    return pd.DatetimeIndex(ts)


def _legacy_recursive_point_forecast(
    model,
    metadata: dict,
    recent_values: np.ndarray,
    horizon_hours: int,
) -> np.ndarray:
    lookback = int(metadata["lookback_hours"])
    mean = float(metadata["mean"])
    std = float(metadata["std"])

    if recent_values.shape[0] < lookback:
        raise ValueError(f"Need at least {lookback} recent values, got {recent_values.shape[0]}")

    seq = apply_zscore(recent_values[-lookback:].astype(np.float32), mean, std).reshape(1, lookback, 1)

    preds_norm = []
    for _ in range(horizon_hours):
        next_norm = float(model.predict(seq, verbose=0)[0][0])
        preds_norm.append(next_norm)
        seq = np.append(seq[:, 1:, :], [[[next_norm]]], axis=1)

    return invert_zscore(np.array(preds_norm, dtype=np.float32), mean, std)


def recursive_forecast_bundle_with_loaded_model(
    model,
    metadata: dict,
    recent_values: np.ndarray,
    horizon_hours: int,
    recent_timestamps: list | np.ndarray | pd.Series | None = None,
) -> dict:
    recent = np.asarray(recent_values, dtype=np.float32).reshape(-1)
    feature_mode = resolve_feature_mode(metadata.get("feature_mode") or "univariate_v0")
    lookback = int(metadata["lookback_hours"])
    mean = float(metadata["mean"])
    std = float(metadata["std"])

    if recent.shape[0] < lookback:
        raise ValueError(f"Need at least {lookback} recent values, got {recent.shape[0]}")

    if feature_mode == "univariate_v0":
        point_forecast = _legacy_recursive_point_forecast(model, metadata, recent, horizon_hours)
        quantiles = calibrated_quantile_forecast(point_forecast, recent, metadata.get("uncertainty"))
        return {
            "point_forecast_m": point_forecast,
            "p10_m": quantiles["p10"],
            "p50_m": quantiles["p50"],
            "p90_m": quantiles["p90"],
            "uncertainty": quantiles["calibration"],
            "feature_mode": feature_mode,
        }

    history_values = recent.astype(np.float32).tolist()
    history_timestamps = list(_coerce_recent_timestamps(recent_timestamps, recent))
    preds_norm: list[float] = []

    for _ in range(horizon_hours):
        normalized_history = apply_zscore(np.array(history_values, dtype=np.float32), mean, std)
        feature_frame = build_feature_frame(
            values_normalized=normalized_history,
            timestamps=history_timestamps,
            feature_mode=feature_mode,
        )
        seq = feature_frame.iloc[-lookback:].to_numpy(dtype=np.float32).reshape(1, lookback, feature_frame.shape[1])
        next_norm = float(model.predict(seq, verbose=0)[0][0])
        preds_norm.append(next_norm)

        next_raw = float(invert_zscore(np.array([next_norm], dtype=np.float32), mean, std)[0])
        history_values.append(next_raw)
        history_timestamps.append(history_timestamps[-1] + pd.Timedelta(hours=1))

    point_forecast = invert_zscore(np.array(preds_norm, dtype=np.float32), mean, std)
    quantiles = calibrated_quantile_forecast(point_forecast, recent, metadata.get("uncertainty"))
    return {
        "point_forecast_m": point_forecast,
        "p10_m": quantiles["p10"],
        "p50_m": quantiles["p50"],
        "p90_m": quantiles["p90"],
        "uncertainty": quantiles["calibration"],
        "feature_mode": feature_mode,
    }


def recursive_forecast_with_loaded_model(
    model,
    metadata: dict,
    recent_values: np.ndarray,
    horizon_hours: int,
    recent_timestamps: list | np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    bundle = recursive_forecast_bundle_with_loaded_model(
        model=model,
        metadata=metadata,
        recent_values=recent_values,
        horizon_hours=horizon_hours,
        recent_timestamps=recent_timestamps,
    )
    return bundle["point_forecast_m"]


def recursive_forecast_bundle(
    model_path: str,
    metadata_path: str,
    recent_values: np.ndarray,
    horizon_hours: int,
    recent_timestamps: list | np.ndarray | pd.Series | None = None,
) -> dict:
    metadata = load_metadata(Path(metadata_path))
    model = load_model(model_path, compile=False)
    return recursive_forecast_bundle_with_loaded_model(
        model=model,
        metadata=metadata,
        recent_values=recent_values,
        horizon_hours=horizon_hours,
        recent_timestamps=recent_timestamps,
    )


def recursive_forecast(
    model_path: str,
    metadata_path: str,
    recent_values: np.ndarray,
    horizon_hours: int,
    recent_timestamps: list | np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    bundle = recursive_forecast_bundle(
        model_path=model_path,
        metadata_path=metadata_path,
        recent_values=recent_values,
        horizon_hours=horizon_hours,
        recent_timestamps=recent_timestamps,
    )
    return bundle["point_forecast_m"]


def main():
    parser = argparse.ArgumentParser(description="Recursive multi-step sea level forecast.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--recent", required=True, help="Comma-separated recent values")
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    recent = np.array([float(v) for v in args.recent.split(",")], dtype=np.float32)
    bundle = recursive_forecast_bundle(args.model, args.metadata, recent, args.horizon)
    print(
        {
            "p10_m": bundle["p10_m"].tolist(),
            "p50_m": bundle["p50_m"].tolist(),
            "p90_m": bundle["p90_m"].tolist(),
            "feature_mode": bundle["feature_mode"],
            "uncertainty": bundle["uncertainty"],
        }
    )


if __name__ == "__main__":
    main()

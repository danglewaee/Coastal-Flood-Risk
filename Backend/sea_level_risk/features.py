from __future__ import annotations

import numpy as np
import pandas as pd

from .exogenous import (
    align_exogenous_to_timestamps,
    resolve_driver_columns,
    standardize_exogenous_frame,
)


FEATURE_SET_NAMES = {
    "univariate_v0": ["sea_level_z"],
    "multivariate_v1": [
        "sea_level_z",
        "delta_1h_z",
        "roll_mean_3h_z",
        "roll_mean_6h_z",
        "roll_std_6h_z",
        "hour_sin",
        "hour_cos",
        "tide12_sin",
        "tide12_cos",
    ],
    "multivariate_v2": [
        "sea_level_z",
        "delta_1h_z",
        "roll_mean_3h_z",
        "roll_mean_6h_z",
        "roll_std_6h_z",
        "hour_sin",
        "hour_cos",
        "tide12_sin",
        "tide12_cos",
    ],
}


def resolve_feature_mode(feature_mode: str | None) -> str:
    if not feature_mode:
        return "univariate_v0"
    mode = feature_mode.strip().lower()
    if mode not in FEATURE_SET_NAMES:
        raise ValueError(f"Unsupported feature_mode '{feature_mode}'. Available: {list(FEATURE_SET_NAMES.keys())}")
    return mode


def _coerce_hourly_timestamps(timestamps: pd.Series | list | np.ndarray | None, length: int) -> pd.DatetimeIndex:
    if timestamps is None:
        return pd.date_range("2000-01-01", periods=length, freq="1h", tz="UTC")

    ts = pd.to_datetime(pd.Series(timestamps), errors="coerce", utc=True)
    if ts.shape[0] != length or ts.isna().all():
        return pd.date_range("2000-01-01", periods=length, freq="1h", tz="UTC")

    if ts.isna().any():
        first_valid = ts.dropna().iloc[0]
        return pd.date_range(first_valid, periods=length, freq="1h", tz="UTC")

    return pd.DatetimeIndex(ts)


def resolve_feature_names(
    feature_mode: str,
    driver_columns: str | list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    mode = resolve_feature_mode(feature_mode)
    if mode != "multivariate_v2":
        return list(FEATURE_SET_NAMES[mode])
    return [*FEATURE_SET_NAMES["multivariate_v2"], *resolve_driver_columns(driver_columns)]


def build_feature_bundle(
    values_normalized: np.ndarray,
    timestamps: pd.Series | list | np.ndarray | None,
    feature_mode: str = "multivariate_v1",
    exogenous_frame: pd.DataFrame | None = None,
    driver_columns: str | list[str] | tuple[str, ...] | None = None,
    exogenous_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    mode = resolve_feature_mode(feature_mode)
    values = np.asarray(values_normalized, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot build features from an empty series.")

    frame = pd.DataFrame({"sea_level_z": values})
    if mode == "univariate_v0":
        feature_names = resolve_feature_names(mode)
        return frame[feature_names].astype(np.float32), {
            "feature_mode": mode,
            "feature_names": feature_names,
            "exogenous": None,
        }

    ts = _coerce_hourly_timestamps(timestamps, len(values))
    series = pd.Series(values, dtype=np.float32)

    frame["delta_1h_z"] = series.diff().fillna(0.0)
    frame["roll_mean_3h_z"] = series.rolling(window=3, min_periods=1).mean()
    frame["roll_mean_6h_z"] = series.rolling(window=6, min_periods=1).mean()
    frame["roll_std_6h_z"] = series.rolling(window=6, min_periods=1).std().fillna(0.0)

    hours = ts.hour + (ts.minute / 60.0)
    frame["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)

    elapsed_hours = np.arange(len(values), dtype=np.float32)
    tidal_period = 12.42
    frame["tide12_sin"] = np.sin(2.0 * np.pi * elapsed_hours / tidal_period)
    frame["tide12_cos"] = np.cos(2.0 * np.pi * elapsed_hours / tidal_period)

    if mode == "multivariate_v1":
        feature_names = resolve_feature_names(mode)
        return frame[feature_names].astype(np.float32), {
            "feature_mode": mode,
            "feature_names": feature_names,
            "exogenous": None,
        }

    aligned_exogenous = align_exogenous_to_timestamps(
        ts,
        exogenous_frame=exogenous_frame,
        driver_columns=driver_columns if driver_columns is not None else (exogenous_stats or {}).get("columns"),
    )
    resolved_driver_columns = resolve_driver_columns(
        driver_columns if driver_columns is not None else (exogenous_stats or {}).get("columns")
    )
    standardized_drivers, driver_stats = standardize_exogenous_frame(
        aligned_exogenous[resolved_driver_columns],
        driver_columns=resolved_driver_columns,
        stats=exogenous_stats,
    )
    frame = pd.concat([frame.reset_index(drop=True), standardized_drivers.reset_index(drop=True)], axis=1)
    feature_names = resolve_feature_names(mode, resolved_driver_columns)
    return frame[feature_names].astype(np.float32), {
        "feature_mode": mode,
        "feature_names": feature_names,
        "exogenous": {
            "driver_columns": resolved_driver_columns,
            "stats": driver_stats,
        },
    }


def build_feature_frame(
    values_normalized: np.ndarray,
    timestamps: pd.Series | list | np.ndarray | None,
    feature_mode: str = "multivariate_v1",
    exogenous_frame: pd.DataFrame | None = None,
    driver_columns: str | list[str] | tuple[str, ...] | None = None,
    exogenous_stats: dict | None = None,
) -> pd.DataFrame:
    frame, _ = build_feature_bundle(
        values_normalized=values_normalized,
        timestamps=timestamps,
        feature_mode=feature_mode,
        exogenous_frame=exogenous_frame,
        driver_columns=driver_columns,
        exogenous_stats=exogenous_stats,
    )
    return frame

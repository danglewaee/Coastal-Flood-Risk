from __future__ import annotations

import numpy as np
import pandas as pd

from .uncertainty import calibrated_quantile_forecast


def tide_persistence_forecast_from_frame(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
    if df.empty:
        raise ValueError("Need non-empty dataframe for tide persistence forecast.")

    series = df.set_index("timestamp")["sea_level"].sort_index().astype(float)
    series = series.resample("1h").mean().interpolate(limit=6, limit_direction="both").dropna()
    if series.shape[0] < 6:
        raise ValueError(f"Need at least 6 hourly observations for tide persistence forecast, got {series.shape[0]}")

    work = series.copy()
    preds: list[float] = []
    for _ in range(horizon_hours):
        candidates: list[float] = [float(work.iloc[-1])]
        weights: list[float] = [0.45]

        if len(work) >= 6:
            candidates.append(float(work.iloc[-6]))
            weights.append(0.15)
        if len(work) >= 12:
            candidates.append(float(work.iloc[-12]))
            weights.append(0.25)
        if len(work) >= 24:
            candidates.append(float(work.iloc[-24]))
            weights.append(0.15)

        pred = float(np.average(np.array(candidates, dtype=np.float32), weights=np.array(weights, dtype=np.float32)))
        if len(work) >= 3:
            short_slope = float((work.iloc[-1] - work.iloc[-3]) / 3.0)
            pred += float(np.clip(short_slope, -0.05, 0.05) * 0.5)

        preds.append(pred)
        next_ts = work.index[-1] + pd.Timedelta(hours=1)
        work = pd.concat([work, pd.Series([pred], index=[next_ts])])

    return np.array(preds, dtype=np.float32)


def tide_persistence_forecast_bundle_from_frame(df: pd.DataFrame, horizon_hours: int) -> dict:
    point_forecast = tide_persistence_forecast_from_frame(df, horizon_hours=horizon_hours)
    recent = df["sea_level"].to_numpy(dtype=np.float32)
    quantiles = calibrated_quantile_forecast(point_forecast, recent_values=recent, calibration=None)
    return {
        "point_forecast_m": point_forecast,
        "p10_m": quantiles["p10"],
        "p50_m": quantiles["p50"],
        "p90_m": quantiles["p90"],
        "uncertainty": quantiles["calibration"],
        "feature_mode": "baseline_tide_persistence",
    }

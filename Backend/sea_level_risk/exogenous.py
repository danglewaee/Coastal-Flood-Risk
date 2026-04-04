from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


DEFAULT_DRIVER_COLUMNS = ["wind_speed", "air_pressure", "precipitation", "river_discharge"]
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HOURLY_VARS = {
    "wind_speed_10m": "wind_speed",
    "pressure_msl": "air_pressure",
    "precipitation": "precipitation",
}


def resolve_driver_columns(driver_columns: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if driver_columns is None:
        return list(DEFAULT_DRIVER_COLUMNS)
    if isinstance(driver_columns, str):
        cols = [part.strip() for part in driver_columns.split(",") if part.strip()]
        return cols or list(DEFAULT_DRIVER_COLUMNS)
    cols = [str(part).strip() for part in driver_columns if str(part).strip()]
    return cols or list(DEFAULT_DRIVER_COLUMNS)


def _request_open_meteo_hourly(
    api_url: str,
    *,
    latitude: float,
    longitude: float,
    hourly_vars: dict[str, str],
    extra_params: dict | None = None,
    timeout: int = 90,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly_vars.keys()),
        "timezone": "UTC",
    }
    if extra_params:
        params.update(extra_params)

    response = requests.get(api_url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly") or {}
    timestamps = hourly.get("time")
    if not timestamps:
        raise ValueError(f"Open-Meteo response from '{api_url}' did not include hourly timestamps.")

    frame = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, errors="coerce", utc=True)})
    for source_name, target_name in hourly_vars.items():
        values = hourly.get(source_name)
        if values is None:
            raise ValueError(f"Open-Meteo response is missing hourly variable '{source_name}'.")
        frame[target_name] = pd.to_numeric(pd.Series(values), errors="coerce")
    return frame.dropna(subset=["timestamp"]).reset_index(drop=True)


def load_exogenous_series(
    csv_path: str,
    *,
    time_col: str = "timestamp",
    driver_columns: str | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Exogenous CSV does not exist: {path}")

    columns = resolve_driver_columns(driver_columns)
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(f"Exogenous CSV '{csv_path}' is missing time column '{time_col}'.")

    available = [col for col in columns if col in df.columns]
    if not available:
        raise ValueError(
            f"Exogenous CSV '{csv_path}' does not contain any requested driver columns. "
            f"Requested={columns}, available={list(df.columns)}"
        )

    work = df[[time_col, *available]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    work = work.dropna(subset=[time_col]).sort_values(time_col)
    for col in available:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    series = work.set_index(time_col)[available].resample("1h").mean().interpolate(limit=6, limit_direction="both")
    series = series.dropna(how="all").reset_index().rename(columns={time_col: "timestamp"})
    return series


def align_exogenous_to_timestamps(
    timestamps: pd.Series | list | np.ndarray | pd.DatetimeIndex,
    exogenous_frame: pd.DataFrame | None,
    *,
    driver_columns: str | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    columns = resolve_driver_columns(driver_columns)
    ts = pd.to_datetime(pd.Series(timestamps), errors="coerce", utc=True)
    aligned = pd.DataFrame({"timestamp": ts})
    if exogenous_frame is None or exogenous_frame.empty:
        for col in columns:
            aligned[col] = 0.0
        return aligned[["timestamp", *columns]]

    if "timestamp" not in exogenous_frame.columns:
        raise ValueError("Aligned exogenous frame requires a 'timestamp' column.")

    available = [col for col in columns if col in exogenous_frame.columns]
    merge_ready = exogenous_frame[["timestamp", *available]].copy()
    merge_ready["timestamp"] = pd.to_datetime(merge_ready["timestamp"], errors="coerce", utc=True)
    merge_ready = merge_ready.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    out = aligned.merge(merge_ready, on="timestamp", how="left").sort_values("timestamp")
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").interpolate(limit=6, limit_direction="both").ffill().bfill().fillna(0.0)
    return out[["timestamp", *columns]]


def standardize_exogenous_frame(
    exogenous_frame: pd.DataFrame,
    *,
    driver_columns: str | list[str] | tuple[str, ...] | None = None,
    stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    columns = resolve_driver_columns(driver_columns)
    if exogenous_frame.empty:
        out = pd.DataFrame({col: np.zeros(0, dtype=np.float32) for col in columns})
        return out, {"columns": columns, "means": {col: 0.0 for col in columns}, "stds": {col: 1.0 for col in columns}}

    work = exogenous_frame.copy()
    means = {}
    stds = {}

    if stats is None:
        source_means = None
        source_stds = None
    else:
        source_means = stats.get("means") or {}
        source_stds = stats.get("stds") or {}

    out = pd.DataFrame(index=work.index)
    for col in columns:
        values = pd.to_numeric(work[col], errors="coerce") if col in work.columns else pd.Series(np.zeros(len(work)), index=work.index)
        values = values.astype(float).ffill().bfill().fillna(0.0)
        mean = float(source_means[col]) if source_means and col in source_means else float(values.mean())
        std = float(source_stds[col]) if source_stds and col in source_stds else float(values.std())
        if std == 0.0:
            std = 1.0
        out[col] = ((values - mean) / std).astype(np.float32)
        means[col] = mean
        stds[col] = std

    return out[columns], {"columns": columns, "means": means, "stds": stds}


def fetch_runtime_exogenous_forecast(
    *,
    latitude: float,
    longitude: float,
    lookback_hours: int,
    horizon_hours: int,
    driver_columns: str | list[str] | tuple[str, ...] | None = None,
    timezone_name: str = "UTC",
    source_label: str = "Open-Meteo Forecast API",
) -> tuple[pd.DataFrame, dict]:
    columns = resolve_driver_columns(driver_columns)
    hourly_vars = {source: target for source, target in OPEN_METEO_HOURLY_VARS.items() if target in columns}
    if not hourly_vars:
        raise ValueError("Runtime exogenous forecast requested without any supported weather driver columns.")

    frame = _request_open_meteo_hourly(
        FORECAST_API_URL,
        latitude=latitude,
        longitude=longitude,
        hourly_vars=hourly_vars,
        extra_params={
            "past_hours": max(int(lookback_hours), 24),
            "forecast_hours": max(int(horizon_hours), 1),
        },
        timeout=60,
    )
    for col in columns:
        if col not in frame.columns:
            frame[col] = 0.0
    frame = frame[["timestamp", *columns]].sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    metadata = {
        "mode": "live_forecast_feed",
        "source": source_label,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone_name,
        "driver_columns": columns,
        "weather_variables": list(hourly_vars.keys()),
        "lookback_hours": int(lookback_hours),
        "forecast_hours": int(horizon_hours),
        "rows": int(frame.shape[0]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Open-Meteo forecast feed is used to provide recent and future hourly weather drivers for multivariate_v2 runtime inference.",
            "river_discharge remains zero-filled until a live discharge source is integrated.",
        ],
    }
    return frame, metadata

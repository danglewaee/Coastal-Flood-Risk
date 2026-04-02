from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .city_registry import load_city_registry
from .data_utils import load_metadata, load_series
from .forecast import recursive_forecast_bundle_with_loaded_model
from .forecast_baselines import tide_persistence_forecast_bundle_from_frame
from .model_registry import resolve_city_model


DEFAULT_BACKTEST_ROOT = Path("Backend/sea_level_risk/outputs/backtests")
DEFAULT_GLOBAL_MODEL_PATH = Path("Backend/sea_level_risk/outputs/sea_level_axial_lstm.keras")
DEFAULT_GLOBAL_METADATA_PATH = Path("Backend/sea_level_risk/outputs/metadata.json")


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _safe_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_pred - y_true))


def _resolve_source_csv(city_key: str, cfg: dict, csv_path: str | None) -> tuple[Path, str]:
    if csv_path:
        candidate = Path(csv_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Backtest CSV does not exist: {candidate}")
        return candidate, "user_csv"

    model_spec = resolve_city_model(city_key)
    if model_spec:
        metadata = load_metadata(Path(model_spec["metadata_path"]))
        training_source = metadata.get("training_source") or {}
        source_csv = training_source.get("source_csv")
        if source_csv:
            candidate = Path(source_csv)
            if candidate.exists():
                return candidate, "model_training_source"

    candidate = Path("data") / f"{city_key}_hourly.csv"
    if candidate.exists():
        return candidate, "default_hourly_cache"

    provider = cfg.get("provider", "unknown")
    raise FileNotFoundError(
        f"No local hourly CSV found for city '{city_key}' (provider={provider}). "
        "Provide --csv or download/cache hourly history first."
    )


def _resolve_model_spec(city_key: str, cfg: dict) -> dict | None:
    city_spec = resolve_city_model(city_key)
    if city_spec is not None:
        return city_spec

    if cfg.get("forecast_mode") == "model_recursive":
        if DEFAULT_GLOBAL_MODEL_PATH.exists() and DEFAULT_GLOBAL_METADATA_PATH.exists():
            return {
                "city": city_key,
                "model_path": str(DEFAULT_GLOBAL_MODEL_PATH),
                "metadata_path": str(DEFAULT_GLOBAL_METADATA_PATH),
                "model_type": "global_default_model",
                "model_dir": str(DEFAULT_GLOBAL_MODEL_PATH.parent),
            }
    return None


def _load_hourly_frame(csv_path: Path, time_col: str, value_col: str) -> pd.DataFrame:
    df = load_series(str(csv_path), time_col=time_col, value_col=value_col)
    if time_col not in df.columns:
        raise ValueError(f"Hourly backtest requires timestamp column '{time_col}' in {csv_path}.")

    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    work = pd.DataFrame({"timestamp": ts, "sea_level": pd.to_numeric(df[value_col], errors="coerce")})
    work = work.dropna(subset=["timestamp", "sea_level"]).sort_values("timestamp")
    series = (
        work.set_index("timestamp")["sea_level"]
        .resample("1h")
        .mean()
        .interpolate(limit=6, limit_direction="both")
        .dropna()
    )
    frame = series.reset_index().rename(columns={"index": "timestamp", "sea_level": "sea_level"})
    if frame.shape[0] < 48:
        raise ValueError(f"Need at least 48 hourly rows for backtest, got {frame.shape[0]} from {csv_path}.")
    return frame


def _build_origin_indices(
    frame_len: int,
    lookback_hours: int,
    horizon_hours: int,
    step_hours: int,
    eval_window_hours: int | None,
    max_windows: int | None,
) -> list[int]:
    last_origin = frame_len - horizon_hours - 1
    if last_origin < lookback_hours - 1:
        raise ValueError(
            f"Not enough hourly rows ({frame_len}) for lookback={lookback_hours} and horizon={horizon_hours}."
        )

    first_origin = lookback_hours - 1
    if eval_window_hours is not None:
        earliest_target_idx = max(lookback_hours, frame_len - int(eval_window_hours))
        first_origin = max(first_origin, earliest_target_idx - 1)

    origins = list(range(first_origin, last_origin + 1, max(int(step_hours), 1)))
    if max_windows is not None and len(origins) > int(max_windows):
        origins = origins[-int(max_windows) :]
    if not origins:
        raise ValueError("Backtest produced zero rolling windows. Increase eval_window_hours or reduce lookback/horizon.")
    return origins


def _window_metrics(group: pd.DataFrame) -> pd.Series:
    y_true = group["y_true_m"].to_numpy(dtype=np.float32)
    y_pred = group["y_pred_p50_m"].to_numpy(dtype=np.float32)
    true_peak_step = int(np.argmax(y_true) + 1)
    pred_peak_step = int(np.argmax(y_pred) + 1)
    return pd.Series(
        {
            "true_peak_m": float(np.max(y_true)),
            "pred_peak_m": float(np.max(y_pred)),
            "peak_abs_error_m": float(abs(np.max(y_pred) - np.max(y_true))),
            "peak_timing_abs_error_h": float(abs(pred_peak_step - true_peak_step)),
        }
    )


def _aggregate_summary(
    records_df: pd.DataFrame,
    *,
    city: str,
    display_name: str,
    forecaster: str,
    horizon_hours: int,
    step_hours: int,
    lookback_hours: int,
    eval_window_hours: int | None,
    source_csv: str,
    model_info: dict | None,
    high_water_quantile: float,
) -> dict:
    y_true = records_df["y_true_m"].to_numpy(dtype=np.float32)
    y_pred = records_df["y_pred_p50_m"].to_numpy(dtype=np.float32)

    summary = {
        "city": city,
        "display_name": display_name,
        "forecaster": forecaster,
        "source_csv": source_csv,
        "lookback_hours": int(lookback_hours),
        "horizon_hours": int(horizon_hours),
        "step_hours": int(step_hours),
        "eval_window_hours": int(eval_window_hours) if eval_window_hours is not None else None,
        "n_windows": int(records_df["origin_timestamp"].nunique()),
        "n_forecast_points": int(records_df.shape[0]),
        "mae_m": _safe_mae(y_true, y_pred),
        "rmse_m": _safe_rmse(y_true, y_pred),
        "bias_m": _safe_bias(y_true, y_pred),
        "start_target_utc": str(records_df["target_timestamp"].min()),
        "end_target_utc": str(records_df["target_timestamp"].max()),
    }

    threshold = float(np.quantile(y_true, high_water_quantile))
    mask = y_true >= threshold
    summary.update(
        {
            "high_water_quantile": float(high_water_quantile),
            "high_water_threshold_m": threshold,
            "high_water_count": int(np.count_nonzero(mask)),
            "high_water_mae_m": _safe_mae(y_true[mask], y_pred[mask]),
            "high_water_rmse_m": _safe_rmse(y_true[mask], y_pred[mask]),
        }
    )

    if {"y_pred_p10_m", "y_pred_p90_m"} <= set(records_df.columns):
        low = records_df["y_pred_p10_m"].to_numpy(dtype=np.float32)
        high = records_df["y_pred_p90_m"].to_numpy(dtype=np.float32)
        coverage = np.logical_and(y_true >= low, y_true <= high)
        summary["p10_p90_coverage"] = float(np.mean(coverage))
        summary["mean_interval_width_m"] = float(np.mean(high - low))
    else:
        summary["p10_p90_coverage"] = None
        summary["mean_interval_width_m"] = None

    peak_df = (
        records_df.groupby("origin_timestamp", group_keys=False)[["y_true_m", "y_pred_p50_m"]]
        .apply(_window_metrics)
        .reset_index(drop=True)
    )
    summary["peak_level_mae_m"] = float(peak_df["peak_abs_error_m"].mean()) if not peak_df.empty else float("nan")
    summary["peak_level_rmse_m"] = (
        float(np.sqrt(np.mean(np.square(peak_df["peak_abs_error_m"].to_numpy(dtype=np.float32)))))
        if not peak_df.empty
        else float("nan")
    )
    summary["peak_timing_mae_h"] = (
        float(peak_df["peak_timing_abs_error_h"].mean()) if not peak_df.empty else float("nan")
    )

    if model_info:
        summary["model"] = model_info

    return summary


def _horizon_metrics(records_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon_step, group in records_df.groupby("horizon_step"):
        y_true = group["y_true_m"].to_numpy(dtype=np.float32)
        y_pred = group["y_pred_p50_m"].to_numpy(dtype=np.float32)
        row = {
            "horizon_step": int(horizon_step),
            "count": int(group.shape[0]),
            "mae_m": _safe_mae(y_true, y_pred),
            "rmse_m": _safe_rmse(y_true, y_pred),
            "bias_m": _safe_bias(y_true, y_pred),
        }
        if {"y_pred_p10_m", "y_pred_p90_m"} <= set(group.columns):
            low = group["y_pred_p10_m"].to_numpy(dtype=np.float32)
            high = group["y_pred_p90_m"].to_numpy(dtype=np.float32)
            row["p10_p90_coverage"] = float(np.mean(np.logical_and(y_true >= low, y_true <= high)))
            row["mean_interval_width_m"] = float(np.mean(high - low))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("horizon_step").reset_index(drop=True)


def _backtest_baseline(
    frame: pd.DataFrame,
    origins: list[int],
    horizon_hours: int,
    lookback_hours: int,
) -> pd.DataFrame:
    records: list[dict] = []
    for end_idx in origins:
        recent = frame.iloc[end_idx - lookback_hours + 1 : end_idx + 1].copy()
        actual = frame.iloc[end_idx + 1 : end_idx + 1 + horizon_hours].copy()
        bundle = tide_persistence_forecast_bundle_from_frame(recent[["timestamp", "sea_level"]], horizon_hours)
        for step_idx in range(horizon_hours):
            records.append(
                {
                    "forecaster": "tide_persistence_baseline",
                    "origin_timestamp": str(recent["timestamp"].iloc[-1]),
                    "target_timestamp": str(actual["timestamp"].iloc[step_idx]),
                    "horizon_step": int(step_idx + 1),
                    "y_true_m": float(actual["sea_level"].iloc[step_idx]),
                    "y_pred_p10_m": float(bundle["p10_m"][step_idx]),
                    "y_pred_p50_m": float(bundle["p50_m"][step_idx]),
                    "y_pred_p90_m": float(bundle["p90_m"][step_idx]),
                }
            )
    return pd.DataFrame(records)


def _backtest_model(
    frame: pd.DataFrame,
    origins: list[int],
    horizon_hours: int,
    lookback_hours: int,
    model_spec: dict,
) -> tuple[pd.DataFrame, dict]:
    metadata = load_metadata(Path(model_spec["metadata_path"]))
    model = load_model(model_spec["model_path"], compile=False)
    records: list[dict] = []
    for end_idx in origins:
        recent = frame.iloc[end_idx - lookback_hours + 1 : end_idx + 1].copy()
        actual = frame.iloc[end_idx + 1 : end_idx + 1 + horizon_hours].copy()
        bundle = recursive_forecast_bundle_with_loaded_model(
            model=model,
            metadata=metadata,
            recent_values=recent["sea_level"].to_numpy(dtype=np.float32),
            horizon_hours=horizon_hours,
            recent_timestamps=recent["timestamp"].tolist(),
        )
        for step_idx in range(horizon_hours):
            records.append(
                {
                    "forecaster": "city_model",
                    "origin_timestamp": str(recent["timestamp"].iloc[-1]),
                    "target_timestamp": str(actual["timestamp"].iloc[step_idx]),
                    "horizon_step": int(step_idx + 1),
                    "y_true_m": float(actual["sea_level"].iloc[step_idx]),
                    "y_pred_p10_m": float(bundle["p10_m"][step_idx]),
                    "y_pred_p50_m": float(bundle["p50_m"][step_idx]),
                    "y_pred_p90_m": float(bundle["p90_m"][step_idx]),
                }
            )
    model_info = {
        "model_path": model_spec["model_path"],
        "metadata_path": model_spec["metadata_path"],
        "model_type": metadata.get("model_type"),
        "feature_mode": metadata.get("feature_mode"),
        "feature_names": metadata.get("feature_names"),
        "uncertainty_method": (metadata.get("uncertainty") or {}).get("method"),
    }
    return pd.DataFrame(records), model_info


def backtest_city(
    *,
    city_key: str,
    csv_path: str | None = None,
    horizon_hours: int = 6,
    step_hours: int = 6,
    eval_window_hours: int | None = 24 * 30,
    max_windows: int | None = None,
    time_col: str = "timestamp",
    value_col: str = "sea_level",
    lookback_hours: int | None = None,
    include_baseline: bool = True,
    include_model: bool = True,
    high_water_quantile: float = 0.9,
    out_root: str | Path = DEFAULT_BACKTEST_ROOT,
) -> dict:
    registry = load_city_registry()
    city_slug = city_key.strip().lower()
    if city_slug not in registry:
        raise ValueError(f"Unknown city '{city_slug}'.")

    cfg = registry[city_slug]
    source_csv, source_mode = _resolve_source_csv(city_slug, cfg, csv_path)
    frame = _load_hourly_frame(source_csv, time_col=time_col, value_col=value_col)
    model_spec = _resolve_model_spec(city_slug, cfg) if include_model else None

    effective_lookback = int(lookback_hours or 24)
    if model_spec:
        metadata = load_metadata(Path(model_spec["metadata_path"]))
        effective_lookback = int(metadata.get("lookback_hours", effective_lookback))

    origins = _build_origin_indices(
        frame_len=frame.shape[0],
        lookback_hours=effective_lookback,
        horizon_hours=horizon_hours,
        step_hours=step_hours,
        eval_window_hours=eval_window_hours,
        max_windows=max_windows,
    )

    results: list[dict] = []
    all_records: list[pd.DataFrame] = []
    all_horizon_metrics: list[pd.DataFrame] = []

    if include_baseline:
        baseline_df = _backtest_baseline(frame, origins, horizon_hours=horizon_hours, lookback_hours=effective_lookback)
        baseline_summary = _aggregate_summary(
            baseline_df,
            city=city_slug,
            display_name=cfg.get("display_name", city_slug),
            forecaster="tide_persistence_baseline",
            horizon_hours=horizon_hours,
            step_hours=step_hours,
            lookback_hours=effective_lookback,
            eval_window_hours=eval_window_hours,
            source_csv=str(source_csv),
            model_info=None,
            high_water_quantile=high_water_quantile,
        )
        all_records.append(baseline_df)
        baseline_horizon = _horizon_metrics(baseline_df)
        baseline_horizon.insert(0, "forecaster", "tide_persistence_baseline")
        all_horizon_metrics.append(baseline_horizon)
        results.append(baseline_summary)

    if include_model and model_spec:
        model_df, model_info = _backtest_model(
            frame,
            origins,
            horizon_hours=horizon_hours,
            lookback_hours=effective_lookback,
            model_spec=model_spec,
        )
        model_summary = _aggregate_summary(
            model_df,
            city=city_slug,
            display_name=cfg.get("display_name", city_slug),
            forecaster="city_model",
            horizon_hours=horizon_hours,
            step_hours=step_hours,
            lookback_hours=effective_lookback,
            eval_window_hours=eval_window_hours,
            source_csv=str(source_csv),
            model_info=model_info,
            high_water_quantile=high_water_quantile,
        )
        all_records.append(model_df)
        model_horizon = _horizon_metrics(model_df)
        model_horizon.insert(0, "forecaster", "city_model")
        all_horizon_metrics.append(model_horizon)
        results.append(model_summary)

    if not results:
        raise ValueError(f"No forecasters were enabled for city '{city_slug}'.")

    city_out = Path(out_root) / city_slug
    city_out.mkdir(parents=True, exist_ok=True)

    records_df = pd.concat(all_records, ignore_index=True)
    horizon_df = pd.concat(all_horizon_metrics, ignore_index=True)

    records_path = city_out / "window_forecasts.csv"
    horizon_path = city_out / "horizon_metrics.csv"
    summary_path = city_out / "summary.json"

    records_df.to_csv(records_path, index=False)
    horizon_df.to_csv(horizon_path, index=False)

    city_payload = {
        "city": city_slug,
        "display_name": cfg.get("display_name", city_slug),
        "provider": cfg.get("provider"),
        "source_csv": str(source_csv),
        "source_mode": source_mode,
        "lookback_hours": int(effective_lookback),
        "horizon_hours": int(horizon_hours),
        "step_hours": int(step_hours),
        "eval_window_hours": int(eval_window_hours) if eval_window_hours is not None else None,
        "n_windows": int(len(origins)),
        "summaries": results,
        "artifacts": {
            "window_forecasts_csv": str(records_path),
            "horizon_metrics_csv": str(horizon_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(city_payload, indent=2), encoding="utf-8")
    return city_payload


def backtest_cities(
    cities: list[str],
    *,
    csv_path: str | None = None,
    horizon_hours: int,
    step_hours: int,
    eval_window_hours: int | None,
    max_windows: int | None,
    time_col: str,
    value_col: str,
    lookback_hours: int | None,
    include_baseline: bool,
    include_model: bool,
    high_water_quantile: float,
    out_root: str | Path,
) -> dict:
    payloads = []
    for city in cities:
        city_csv = csv_path if len(cities) == 1 else None
        payloads.append(
            backtest_city(
                city_key=city,
                csv_path=city_csv,
                horizon_hours=horizon_hours,
                step_hours=step_hours,
                eval_window_hours=eval_window_hours,
                max_windows=max_windows,
                time_col=time_col,
                value_col=value_col,
                lookback_hours=lookback_hours,
                include_baseline=include_baseline,
                include_model=include_model,
                high_water_quantile=high_water_quantile,
                out_root=out_root,
            )
        )

    aggregate = {
        "cities": payloads,
        "n_cities": len(payloads),
    }
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    aggregate_path = root / "run_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    aggregate["aggregate_summary_json"] = str(aggregate_path)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Rolling backtest suite for coastal water-level forecasters.")
    parser.add_argument("--cities", nargs="*", default=None, help="City keys to backtest. Defaults to all NOAA cities.")
    parser.add_argument("--all-noaa", action="store_true", help="Backtest every NOAA city in the registry.")
    parser.add_argument("--csv", default=None, help="Optional CSV override for single-city backtests.")
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--value-col", default="sea_level")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--eval-window-hours", type=int, default=24 * 30)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--lookback-hours", type=int, default=None)
    parser.add_argument("--high-water-quantile", type=float, default=0.9)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--out-root", default=str(DEFAULT_BACKTEST_ROOT))
    args = parser.parse_args()

    registry = load_city_registry()
    if args.all_noaa:
        cities = [city for city, cfg in registry.items() if cfg.get("provider") == "noaa"]
    elif args.cities:
        cities = [city.strip().lower() for city in args.cities]
    else:
        cities = [city for city, cfg in registry.items() if cfg.get("provider") == "noaa"]

    if args.csv and len(cities) != 1:
        raise ValueError("--csv override only works with a single city.")

    result = backtest_cities(
        cities=cities,
        csv_path=args.csv,
        horizon_hours=args.horizon,
        step_hours=args.step_hours,
        eval_window_hours=args.eval_window_hours,
        max_windows=args.max_windows,
        time_col=args.time_col,
        value_col=args.value_col,
        lookback_hours=args.lookback_hours,
        include_baseline=not args.skip_baseline,
        include_model=not args.skip_model,
        high_water_quantile=args.high_water_quantile,
        out_root=args.out_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

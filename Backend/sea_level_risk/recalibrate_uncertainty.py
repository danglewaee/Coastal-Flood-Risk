from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest import backtest_city
from .city_registry import load_city_registry
from .model_registry import DEFAULT_MODELS_ROOT, resolve_city_model
from .uncertainty import build_horizon_quantile_calibration


DEFAULT_RECALIBRATION_ROOT = Path("Backend/sea_level_risk/outputs/recalibration_runs")
DEFAULT_GLOBAL_METADATA_PATH = Path("Backend/sea_level_risk/outputs/metadata.json")


def _resolve_metadata_path(city_key: str, cfg: dict, *, models_root: str | Path = DEFAULT_MODELS_ROOT) -> Path:
    city_spec = resolve_city_model(city_key, models_root=models_root)
    if city_spec is not None:
        return Path(city_spec["metadata_path"])

    if cfg.get("forecast_mode") == "model_recursive" and DEFAULT_GLOBAL_METADATA_PATH.exists():
        return DEFAULT_GLOBAL_METADATA_PATH

    raise FileNotFoundError(f"No model metadata found for city '{city_key}'.")


def recalibrate_city_uncertainty(
    *,
    city_key: str,
    csv_path: str | None = None,
    horizon_hours: int = 6,
    step_hours: int = 6,
    eval_window_hours: int | None = 24 * 60,
    max_windows: int | None = None,
    time_col: str = "timestamp",
    value_col: str = "sea_level",
    lookback_hours: int | None = None,
    central_interval_coverage: float = 0.80,
    out_root: str | Path = DEFAULT_RECALIBRATION_ROOT,
    models_root: str | Path = DEFAULT_MODELS_ROOT,
) -> dict:
    registry = load_city_registry()
    city_slug = city_key.strip().lower()
    if city_slug not in registry:
        raise ValueError(f"Unknown city '{city_slug}'.")

    city_payload = backtest_city(
        city_key=city_slug,
        csv_path=csv_path,
        horizon_hours=horizon_hours,
        step_hours=step_hours,
        eval_window_hours=eval_window_hours,
        max_windows=max_windows,
        time_col=time_col,
        value_col=value_col,
        lookback_hours=lookback_hours,
        include_baseline=False,
        include_model=True,
        high_water_quantile=0.9,
        out_root=out_root,
        models_root=models_root,
    )

    records_path = Path(city_payload["artifacts"]["window_forecasts_csv"])
    records_df = pd.read_csv(records_path)
    model_df = records_df[records_df["forecaster"] == "city_model"].copy()
    if model_df.empty:
        raise ValueError(f"No city model forecasts were produced for city '{city_slug}'.")

    calibration = build_horizon_quantile_calibration(
        model_df,
        central_interval_coverage=central_interval_coverage,
    )

    metadata_path = _resolve_metadata_path(city_slug, registry[city_slug], models_root=models_root)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    metadata["uncertainty"] = calibration
    metadata["uncertainty_recalibrated_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["uncertainty_recalibration"] = {
        "source": "rolling_backtest",
        "city": city_slug,
        "horizon_hours": int(horizon_hours),
        "step_hours": int(step_hours),
        "eval_window_hours": int(eval_window_hours) if eval_window_hours is not None else None,
        "max_windows": int(max_windows) if max_windows is not None else None,
        "central_interval_coverage": float(central_interval_coverage),
        "records_csv": str(records_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "city": city_slug,
        "metadata_path": str(metadata_path),
        "records_csv": str(records_path),
        "calibration_method": calibration["method"],
        "empirical_coverage": calibration["empirical_coverage"],
        "central_interval_coverage_target": calibration["central_interval_coverage_target"],
    }


def main():
    parser = argparse.ArgumentParser(description="Recalibrate forecast uncertainty using rolling backtest residuals.")
    parser.add_argument("--cities", nargs="*", default=None, help="City keys to recalibrate. Defaults to all NOAA cities with models.")
    parser.add_argument("--all-noaa", action="store_true")
    parser.add_argument("--csv", default=None, help="Optional CSV override for a single-city recalibration.")
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--value-col", default="sea_level")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--eval-window-hours", type=int, default=24 * 60)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--lookback-hours", type=int, default=None)
    parser.add_argument("--coverage-target", type=float, default=0.80)
    parser.add_argument("--out-root", default=str(DEFAULT_RECALIBRATION_ROOT))
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT))
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

    results = []
    for city in cities:
        results.append(
            recalibrate_city_uncertainty(
                city_key=city,
                csv_path=args.csv if len(cities) == 1 else None,
                horizon_hours=args.horizon,
                step_hours=args.step_hours,
                eval_window_hours=args.eval_window_hours,
                max_windows=args.max_windows,
                time_col=args.time_col,
                value_col=args.value_col,
                lookback_hours=args.lookback_hours,
                central_interval_coverage=args.coverage_target,
                out_root=args.out_root,
                models_root=args.models_root,
            )
        )

    print(json.dumps({"cities": results}, indent=2))


if __name__ == "__main__":
    main()

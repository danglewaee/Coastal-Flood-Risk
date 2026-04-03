from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .city_registry import load_city_registry
from .config import TrainConfig
from .download_data import download_noaa_hourly
from .train import train_model


def _validate_hourly_series(csv_path: str, time_col: str | None, value_col: str) -> dict:
    df = pd.read_csv(csv_path)
    if value_col not in df.columns:
        raise ValueError(f"CSV '{csv_path}' is missing value column '{value_col}'.")

    summary = {
        "rows": int(len(df)),
        "time_col": time_col,
        "value_col": value_col,
        "median_step_hours": None,
    }

    if not time_col or time_col not in df.columns:
        return summary

    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True).dropna().sort_values()
    if ts.shape[0] < 10:
        raise ValueError(f"CSV '{csv_path}' does not contain enough valid timestamps to validate cadence.")

    diffs = ts.diff().dropna().dt.total_seconds() / 3600.0
    if diffs.empty:
        raise ValueError(f"CSV '{csv_path}' does not contain enough valid timestamps to validate cadence.")

    median_step = float(diffs.median())
    summary["median_step_hours"] = median_step
    if median_step > 2.0:
        raise ValueError(
            f"CSV '{csv_path}' appears to be too coarse for the current short-horizon forecast task "
            f"(median step {median_step:.2f} hours). Use hourly gauge history instead."
        )
    return summary


def _prepare_training_csv(
    city_key: str,
    cfg: dict,
    csv_path: str | None,
    begin_date: str,
    end_date: str,
    datum: str,
    time_col: str | None,
    value_col: str,
) -> tuple[str, dict]:
    if csv_path:
        validation = _validate_hourly_series(csv_path, time_col=time_col, value_col=value_col)
        return csv_path, {
            "source_mode": "user_csv",
            "source_csv": csv_path,
            "validation": validation,
        }

    if cfg.get("provider") != "noaa":
        raise ValueError(
            f"City '{city_key}' does not have automatic hourly history download implemented. "
            "Provide an hourly CSV with --csv."
        )

    station_id = cfg.get("station_id")
    if not station_id:
        raise ValueError(f"City '{city_key}' is missing station_id.")

    out_csv = Path("data") / f"{city_key}_hourly.csv"
    download_noaa_hourly(
        station=station_id,
        begin_date=begin_date,
        end_date=end_date,
        out_csv=str(out_csv),
        product="water_level",
        datum=datum,
    )
    validation = _validate_hourly_series(str(out_csv), time_col=time_col, value_col=value_col)
    return str(out_csv), {
        "source_mode": "noaa_download",
        "source_csv": str(out_csv),
        "station_id": station_id,
        "begin_date": begin_date,
        "end_date": end_date,
        "validation": validation,
    }


def train_city_model(
    city_key: str,
    csv_path: str | None,
    begin_date: str,
    end_date: str,
    datum: str,
    model_type: str,
    time_col: str | None,
    value_col: str,
    out_root: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    lookback_hours: int | None = None,
    feature_mode: str | None = None,
    drivers_csv: str | None = None,
    drivers_time_col: str = "timestamp",
    driver_cols: str | None = None,
    reuse_model: bool = False,
) -> dict:
    registry = load_city_registry()
    city_slug = city_key.strip().lower()
    if city_slug not in registry:
        raise ValueError(f"Unknown city '{city_slug}'. Available: {list(registry.keys())}")

    city_cfg = registry[city_slug]
    training_csv, source_info = _prepare_training_csv(
        city_key=city_slug,
        cfg=city_cfg,
        csv_path=csv_path,
        begin_date=begin_date,
        end_date=end_date,
        datum=datum,
        time_col=time_col,
        value_col=value_col,
    )

    output_dir = Path(out_root) / city_slug
    cfg = TrainConfig()
    if epochs is not None:
        cfg = replace(cfg, epochs=int(epochs))
    if batch_size is not None:
        cfg = replace(cfg, batch_size=int(batch_size))
    if lookback_hours is not None:
        cfg = replace(cfg, lookback_hours=int(lookback_hours))
    if feature_mode is not None:
        cfg = replace(cfg, feature_mode=str(feature_mode))

    model_path = output_dir / f"sea_level_{model_type}.keras"
    metadata_path = output_dir / "metadata.json"
    if reuse_model and model_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        metadata.update(
            {
                "city": city_slug,
                "display_name": city_cfg.get("display_name"),
                "provider": city_cfg.get("provider"),
                "provider_label": city_cfg.get("provider_label"),
                "station_ref": city_cfg.get("station_id") or city_cfg.get("station_code"),
                "forecast_task": "hourly_short_horizon_water_level",
                "training_source": source_info,
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {
            "city": city_slug,
            "display_name": city_cfg.get("display_name"),
            "training_csv": training_csv,
            "output_dir": str(output_dir),
            "training_source": source_info,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "model_type": model_type,
            "reused_existing_model": True,
        }

    result = train_model(
        csv_path=training_csv,
        value_col=value_col,
        time_col=time_col,
        output_dir=output_dir,
        cfg=cfg,
        model_type=model_type,
        drivers_csv=drivers_csv,
        drivers_time_col=drivers_time_col,
        driver_columns=driver_cols,
    )

    metadata_path = Path(result["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    metadata.update(
        {
            "city": city_slug,
            "display_name": city_cfg.get("display_name"),
            "provider": city_cfg.get("provider"),
            "provider_label": city_cfg.get("provider_label"),
            "station_ref": city_cfg.get("station_id") or city_cfg.get("station_code"),
            "forecast_task": "hourly_short_horizon_water_level",
            "training_source": source_info,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        **result,
        "city": city_slug,
        "display_name": city_cfg.get("display_name"),
        "training_csv": training_csv,
        "output_dir": str(output_dir),
        "training_source": source_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a city-specific deep-learning model for hourly water-level forecasting.")
    parser.add_argument("--city", required=True, help="City key from city_registry.json")
    parser.add_argument("--csv", default=None, help="Optional hourly CSV. If omitted and city uses NOAA, hourly history is downloaded automatically.")
    parser.add_argument("--begin", default="20100101", help="NOAA download begin date (YYYYMMDD)")
    parser.add_argument("--end", default=pd.Timestamp.utcnow().strftime("%Y%m%d"), help="NOAA download end date (YYYYMMDD)")
    parser.add_argument("--datum", default="MSL")
    parser.add_argument("--value-col", default="sea_level")
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--model-type", default="axial_lstm", choices=["lstm", "temporal_cnn", "axial_lstm"])
    parser.add_argument("--out-root", default="Backend/sea_level_risk/outputs/models")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lookback-hours", type=int, default=None)
    parser.add_argument("--feature-mode", default="multivariate_v1", choices=["univariate_v0", "multivariate_v1", "multivariate_v2"])
    parser.add_argument("--drivers-csv", default=None, help="Optional hourly exogenous driver CSV for multivariate_v2.")
    parser.add_argument("--drivers-time-col", default="timestamp")
    parser.add_argument("--driver-cols", default=None, help="Comma-separated exogenous driver columns.")
    parser.add_argument("--reuse-model", action="store_true")
    args = parser.parse_args()

    result = train_city_model(
        city_key=args.city,
        csv_path=args.csv,
        begin_date=args.begin,
        end_date=args.end,
        datum=args.datum,
        model_type=args.model_type,
        time_col=args.time_col,
        value_col=args.value_col,
        out_root=args.out_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lookback_hours=args.lookback_hours,
        feature_mode=args.feature_mode,
        drivers_csv=args.drivers_csv,
        drivers_time_col=args.drivers_time_col,
        driver_cols=args.driver_cols,
        reuse_model=args.reuse_model,
    )
    print(result)


if __name__ == "__main__":
    main()

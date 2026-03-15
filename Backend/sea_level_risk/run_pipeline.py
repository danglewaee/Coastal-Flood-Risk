import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TrainConfig
from .data_utils import load_series
from .forecast import recursive_forecast
from .gis import compute_exposure, dem_to_flood_polygon, save_summary
from .train import train_model


DEFAULT_SCENARIOS_M = [0.2, 0.5, 1.0]


def run_pipeline(
    csv_path: str,
    dem_path: str,
    value_col: str,
    time_col: str | None,
    output_dir: Path,
    population_layer: str | None,
    infra_layer: str | None,
    horizon_hours: int,
    model_type: str,
    reuse_model: bool,
):
    cfg = TrainConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"sea_level_{model_type}.keras"
    metadata_path = output_dir / "metadata.json"

    if reuse_model and model_path.exists() and metadata_path.exists():
        train_result = {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "model_type": model_type,
            "reused_existing_model": True,
        }
    else:
        train_result = train_model(csv_path, value_col, time_col, output_dir, cfg, model_type=model_type)

    df = load_series(csv_path=csv_path, time_col=time_col, value_col=value_col)
    recent_values = df[value_col].to_numpy(dtype=np.float32)

    predictions = recursive_forecast(
        model_path=train_result["model_path"],
        metadata_path=train_result["metadata_path"],
        recent_values=recent_values,
        horizon_hours=horizon_hours,
    )

    base_level = float(np.max(predictions))

    scenario_rows = []
    for delta in DEFAULT_SCENARIOS_M:
        scenario_level = base_level + delta
        scenario_name = f"plus_{int(delta * 100)}cm"
        flood_geojson = output_dir / f"flood_{scenario_name}.geojson"
        flood_result = dem_to_flood_polygon(dem_path, scenario_level, str(flood_geojson))

        exposures = []
        if population_layer:
            exposures.append(compute_exposure(str(flood_geojson), population_layer, "population"))
        if infra_layer:
            exposures.append(compute_exposure(str(flood_geojson), infra_layer, "infrastructure"))

        summary = {
            "scenario": scenario_name,
            "base_predicted_peak_m": base_level,
            "scenario_water_level_m": scenario_level,
            "flood": flood_result,
            "exposure": exposures,
        }
        save_summary(str(output_dir / f"summary_{scenario_name}.json"), summary)

        row = {
            "scenario": scenario_name,
            "scenario_water_level_m": scenario_level,
            "flood_area_m2": flood_result["flood_area_m2"],
            "flood_ratio": flood_result.get("flood_ratio", 0.0),
            "component_count": flood_result.get("component_count", 0),
            "processing_mode": flood_result.get("processing_mode", "unknown"),
        }
        for e in exposures:
            row[f"{e['layer']}_intersections"] = e["intersections"]
            row[f"{e['layer']}_affected_area_m2"] = e["affected_area_m2"]

        scenario_rows.append(row)

    metrics = {
        "train": train_result,
        "forecast_horizon_hours": horizon_hours,
        "forecast_values_m": predictions.tolist(),
        "peak_prediction_m": base_level,
    }
    save_summary(str(output_dir / "metrics.json"), metrics)

    pd.DataFrame(scenario_rows).to_csv(output_dir / "scenario_summary.csv", index=False)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="End-to-end sea level forecasting + GIS risk mapping pipeline")
    parser.add_argument("--csv", required=True, help="Hourly sea-level CSV")
    parser.add_argument("--dem", required=True, help="DEM raster path (.tif)")
    parser.add_argument("--value-col", default="sea_level")
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--population", default=None, help="Optional population vector layer")
    parser.add_argument("--infra", default=None, help="Optional infrastructure vector layer")
    parser.add_argument("--horizon", type=int, default=6, help="Recursive forecast horizon")
    parser.add_argument("--model-type", default="lstm", choices=["lstm", "temporal_cnn", "axial_lstm"])
    parser.add_argument("--reuse-model", action="store_true", help="Skip training if model + metadata already exist")
    parser.add_argument("--out", default="Backend/sea_level_risk/outputs")
    args = parser.parse_args()

    result = run_pipeline(
        csv_path=args.csv,
        dem_path=args.dem,
        value_col=args.value_col,
        time_col=args.time_col,
        output_dir=Path(args.out),
        population_layer=args.population,
        infra_layer=args.infra,
        horizon_hours=args.horizon,
        model_type=args.model_type,
        reuse_model=args.reuse_model,
    )
    print(result)


if __name__ == "__main__":
    main()

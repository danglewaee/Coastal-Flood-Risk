import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def render_maps(outputs_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    maps_dir = outputs_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    scenario_colors = {
        "plus_20cm": "#f6bd60",
        "plus_50cm": "#f28482",
        "plus_100cm": "#84a59d",
    }

    generated = []
    for scenario in ["plus_20cm", "plus_50cm", "plus_100cm"]:
        geojson_path = outputs_dir / f"flood_{scenario}.geojson"
        if not geojson_path.exists():
            continue

        gdf = gpd.read_file(geojson_path)
        fig, ax = plt.subplots(figsize=(8, 8))

        if gdf.empty:
            ax.text(0.5, 0.5, "No flooded polygon", ha="center", va="center")
            ax.set_axis_off()
        else:
            gdf.plot(ax=ax, color=scenario_colors.get(scenario, "#457b9d"), edgecolor="none", alpha=0.75)
            ax.set_title(f"Flood Risk Map - {scenario.replace('plus_', '+').replace('cm', ' cm')}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        out_png = maps_dir / f"map_{scenario}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close(fig)
        generated.append(str(out_png))

    return generated


def generate_report(outputs_dir: Path) -> Path:
    metrics_path = outputs_dir / "metrics.json"
    scenarios_path = outputs_dir / "scenario_summary.csv"

    if not metrics_path.exists() or not scenarios_path.exists():
        raise FileNotFoundError("Missing metrics.json or scenario_summary.csv")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    scenarios = pd.read_csv(scenarios_path)

    forecast_vals = metrics.get("forecast_values_m", [])
    peak_pred = metrics.get("peak_prediction_m")
    model_type = metrics.get("train", {}).get("model_type", "unknown")

    lines = []
    lines.append("# Sea Level Rise Project - Results Summary")
    lines.append("")
    lines.append("## Forecast")
    lines.append(f"- Model: `{model_type}`")
    lines.append(f"- Forecast horizon: `{metrics.get('forecast_horizon_hours', 'N/A')}` hours")
    lines.append(f"- Peak predicted sea level: `{peak_pred:.6f} m`" if peak_pred is not None else "- Peak predicted sea level: N/A")
    lines.append(f"- Forecast values (m): `{forecast_vals}`")
    lines.append("")
    lines.append("## Scenario Impact")
    lines.append("")
    lines.append("| Scenario | Water level (m) | Flood area (m2) | Flood ratio | Components |")
    lines.append("|---|---:|---:|---:|---:|")

    for _, row in scenarios.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['scenario_water_level_m']:.6f} | {row['flood_area_m2']:.6f} | {row.get('flood_ratio', 0.0):.6f} | {int(row.get('component_count', 0))} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- Scenario levels are computed from forecasted peak + {20, 50, 100} cm.")
    lines.append("- Flood polygons are generated from DEM cells where elevation is above 0 m and below the scenario water level.")
    lines.append("- Only coast-connected flood components are kept, then small polygons are filtered and geometries are cleaned for QGIS/reporting.")

    out_md = outputs_dir / "results_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def main():
    parser = argparse.ArgumentParser(description="Generate map images and report from pipeline outputs")
    parser.add_argument("--out", default="Backend/sea_level_risk/outputs", help="Pipeline output directory")
    args = parser.parse_args()

    outputs_dir = Path(args.out)

    map_files = render_maps(outputs_dir)
    report_path = generate_report(outputs_dir)

    print({"report": str(report_path), "maps": map_files})


if __name__ == "__main__":
    main()

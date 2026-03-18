from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gis import depth_raster_to_flood_polygon


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess hydrodynamic depth raster into flood polygons.")
    parser.add_argument("--city", required=True, help="City key from city_registry.json (e.g., boston)")
    parser.add_argument("--scenario", required=True, help="Scenario name (plus_20cm, plus_50cm, plus_100cm)")
    parser.add_argument("--depth", required=True, help="Path to depth GeoTIFF exported from HEC-RAS or similar")
    parser.add_argument("--dem", default=None, help="Optional DEM path aligned to the depth raster grid")
    parser.add_argument("--out-dir", default=None, help="Output directory for geojson/meta")
    parser.add_argument("--depth-threshold", type=float, default=0.01, help="Depth threshold in meters")
    parser.add_argument("--min-land-elevation", type=float, default=0.0, help="Minimum land elevation for ratio (if DEM provided)")
    parser.add_argument("--min-component-area", type=float, default=2500.0, help="Min component area in m2")
    parser.add_argument("--smooth", type=float, default=10.0, help="Geometry simplification tolerance in meters")
    parser.add_argument("--scenario-water-level", type=float, default=None, help="Optional absolute water level for this scenario (m)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("Backend/sea_level_risk/outputs/realtime") / args.city
    out_dir.mkdir(parents=True, exist_ok=True)
    out_geojson = out_dir / f"flood_{args.scenario}.geojson"

    result = depth_raster_to_flood_polygon(
        depth_path=args.depth,
        out_geojson=str(out_geojson),
        dem_path=args.dem,
        depth_threshold_m=args.depth_threshold,
        min_land_elevation_m=args.min_land_elevation,
        min_component_area_m2=args.min_component_area,
        smooth_tolerance_m=args.smooth,
    )

    meta = {
        "scenario": args.scenario,
        "scenario_water_level_m": args.scenario_water_level,
        "predicted_level_m": args.scenario_water_level,
        "processing_mode": "hydro_model",
        "depth_threshold_m": args.depth_threshold,
        "min_land_elevation_m": args.min_land_elevation,
        "min_component_area_m2": args.min_component_area,
        "smooth_tolerance_m": args.smooth,
        "flood_pixels": result.get("flood_pixels"),
        "land_pixels": result.get("land_pixels"),
        "component_count": result.get("component_count"),
        "flood_ratio": result.get("flood_ratio"),
        "flood_area_m2": result.get("flood_area_m2"),
        "dem_path": args.dem,
        "depth_path": args.depth,
    }

    out_geojson.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd

from .gis import EQUAL_AREA_EPSG


def load_hydro_override(out_dir: Path, scenario_name: str) -> dict | None:
    geojson_path = Path(out_dir) / f"flood_{scenario_name}.geojson"
    meta_path = geojson_path.with_suffix(".meta.json")

    if not geojson_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None

    if meta.get("processing_mode") != "hydro_model":
        return None

    flood_area_m2 = meta.get("flood_area_m2")
    component_count = meta.get("component_count")
    if flood_area_m2 is None or component_count is None:
        gdf = gpd.read_file(geojson_path)
        flood_area_m2 = float(gdf.to_crs(epsg=EQUAL_AREA_EPSG).geometry.area.sum()) if not gdf.empty else 0.0
        component_count = int(len(gdf))

    flood_ratio = meta.get("flood_ratio")
    flood_pixels = meta.get("flood_pixels")
    land_pixels = meta.get("land_pixels")
    if flood_ratio is None:
        flood_ratio = 0.0

    return {
        "scenario": scenario_name,
        "scenario_water_level_m": meta.get("scenario_water_level_m"),
        "predicted_level_m": meta.get("predicted_level_m"),
        "processing_mode": meta.get("processing_mode"),
        "flood_area_m2": float(flood_area_m2),
        "flood_ratio": float(flood_ratio),
        "component_count": int(component_count),
        "flood_pixels": 0 if flood_pixels is None else int(flood_pixels),
        "land_pixels": 0 if land_pixels is None else int(land_pixels),
        "risk_level": meta.get("risk_level"),
        "geojson": str(geojson_path),
    }

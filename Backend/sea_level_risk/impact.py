from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd

from .gis import compute_exposure
from .priority import ensure_hotspots_from_scenarios, hotspot_records_from_gdf


DEFAULT_EXPOSURE_REGISTRY = {}


def load_exposure_registry(path: str = "Backend/sea_level_risk/exposure_registry.json") -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8-sig"))
    return dict(DEFAULT_EXPOSURE_REGISTRY)


def _scenario_hotspot_metrics(hotspot_gdf: gpd.GeoDataFrame, scenario: str) -> dict:
    records = hotspot_records_from_gdf(hotspot_gdf, limit=5, scenario=scenario)
    if not records:
        return {
            "hotspot_count": 0,
            "top_hotspots": [],
            "largest_hotspot_area_m2": None,
            "top3_hotspot_area_m2": None,
        }

    largest = max(float(item.get("area_m2", 0.0)) for item in records)
    top3_area = sum(float(item.get("area_m2", 0.0)) for item in records[:3])
    return {
        "hotspot_count": int(len(records)),
        "top_hotspots": records,
        "largest_hotspot_area_m2": largest,
        "top3_hotspot_area_m2": top3_area,
    }


def build_impact_summaries(
    payload: dict,
    city_key: str | None,
    hotspot_geojson: str,
    hotspot_csv: str,
    exposure_registry: dict | None = None,
) -> dict:
    scenario_records = payload.get("scenarios", [])
    if not scenario_records:
        return {}

    hotspot_gdf = ensure_hotspots_from_scenarios(
        scenario_records=scenario_records,
        out_geojson=hotspot_geojson,
        out_csv=hotspot_csv,
        top_n=20,
    )

    registry = exposure_registry or load_exposure_registry()
    city_layers = registry.get(city_key or "", [])
    summaries = {}

    for scenario in scenario_records:
        scenario_name = scenario.get("scenario")
        metrics = _scenario_hotspot_metrics(hotspot_gdf, scenario_name)
        exposure_rows = []
        for layer_cfg in city_layers:
            layer_path = layer_cfg.get("path")
            layer_name = layer_cfg.get("name", "layer")
            if not layer_path or not Path(layer_path).exists():
                continue
            try:
                exposure_rows.append(compute_exposure(scenario["geojson"], layer_path, layer_name=layer_name))
            except Exception:
                continue

        flood_ratio = scenario.get("flood_ratio")
        summaries[scenario_name] = {
            "scenario": scenario_name,
            "risk_level": scenario.get("risk_level"),
            "flood_ratio_pct": None if flood_ratio is None else float(flood_ratio) * 100.0,
            "flood_area_m2": None if scenario.get("flood_area_m2") is None else float(scenario.get("flood_area_m2")),
            "component_count": int(scenario.get("component_count", 0)),
            "processing_mode": scenario.get("processing_mode"),
            "hotspot_count": metrics["hotspot_count"],
            "top_hotspots": metrics["top_hotspots"],
            "largest_hotspot_area_m2": metrics["largest_hotspot_area_m2"],
            "top3_hotspot_area_m2": metrics["top3_hotspot_area_m2"],
            "exposure_layers_configured": int(len(city_layers)),
            "exposure_layers_available": int(len(exposure_rows)),
            "exposure_summary": exposure_rows,
            "hotspots_geojson": hotspot_geojson,
            "hotspots_csv": hotspot_csv,
        }

    return summaries

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd

from .gis import compute_exposure
from .priority import ensure_hotspots_from_scenarios, hotspot_records_from_gdf


DEFAULT_EXPOSURE_REGISTRY = {}
POINT_METRIC = "points"
LENGTH_METRIC = "length_m"
AREA_METRIC = "area_m2"
PEOPLE_METRIC = "people"

CATEGORY_LABELS = {
    "transport": "Road network",
    "healthcare": "Hospitals & clinics",
    "emergency_response": "Emergency response sites",
    "education": "Schools & campuses",
    "power": "Power substations",
    "critical_services": "Critical facilities",
    "population": "Population exposure",
    "vulnerability": "High social-vulnerability population",
}


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


def _enrich_exposure_row(row: dict, layer_cfg: dict) -> dict:
    metric = str(layer_cfg.get("metric") or "").lower()
    if metric == LENGTH_METRIC:
        affected_value = float(row.get("affected_length_m", 0.0))
        affected_unit = "m"
    elif metric == AREA_METRIC:
        affected_value = float(row.get("affected_area_m2", 0.0))
        affected_unit = "m2"
    elif metric == PEOPLE_METRIC:
        affected_value = float(row.get("affected_weighted_value", 0.0))
        affected_unit = "people"
    else:
        metric = POINT_METRIC
        affected_value = int(row.get("affected_point_count", 0))
        affected_unit = "sites"

    enriched = dict(row)
    enriched["display_name"] = layer_cfg.get("display_name", layer_cfg.get("name", row.get("layer", "layer")))
    enriched["category"] = layer_cfg.get("category", layer_cfg.get("name", row.get("layer", "other")))
    enriched["metric"] = metric
    enriched["affected_value"] = affected_value
    enriched["affected_unit"] = affected_unit
    return enriched


def _rollup_exposure_rows(exposure_rows: list[dict]) -> dict:
    groups: dict[tuple[str, str], dict] = {}
    total_sites = 0
    total_road_length_m = 0.0
    total_population_affected = 0.0
    total_high_vulnerability_population = 0.0
    headline_items: list[str] = []

    for row in exposure_rows:
        category = str(row.get("category") or "other")
        metric = str(row.get("metric") or POINT_METRIC)
        key = (category, metric)
        group = groups.setdefault(
            key,
            {
                "category": category,
                "display_name": CATEGORY_LABELS.get(category, category.replace("_", " ").title()),
                "metric": metric,
                "affected_value": 0.0,
                "affected_unit": row.get("affected_unit"),
                "layers": [],
            },
        )
        group["affected_value"] += float(row.get("affected_value", 0.0))
        group["layers"].append(row.get("display_name") or row.get("layer"))

        if metric == POINT_METRIC:
            total_sites += int(row.get("affected_value", 0))
        if category == "transport" and metric == LENGTH_METRIC:
            total_road_length_m += float(row.get("affected_value", 0.0))
        if category == "population" and metric == PEOPLE_METRIC:
            total_population_affected += float(row.get("affected_value", 0.0))
        if category == "vulnerability" and metric == PEOPLE_METRIC:
            total_high_vulnerability_population += float(row.get("affected_value", 0.0))

    rollup_rows = []
    for group in groups.values():
        value = group["affected_value"]
        if group["metric"] == POINT_METRIC:
            value = int(round(value))
        else:
            value = round(float(value), 1)
        rollup_rows.append(
            {
                "category": group["category"],
                "display_name": group["display_name"],
                "metric": group["metric"],
                "affected_value": value,
                "affected_unit": group["affected_unit"],
                "layers": ", ".join(sorted({item for item in group["layers"] if item})),
            }
        )

    rollup_rows.sort(key=lambda item: float(item["affected_value"]) if item["affected_value"] is not None else 0.0, reverse=True)

    if total_road_length_m > 0.0:
        headline_items.append(f"{total_road_length_m / 1000.0:.1f} km of road network intersect projected flooding")
    if total_sites > 0:
        headline_items.append(f"{total_sites} priority sites intersect projected flooding")
    if total_population_affected > 0.0:
        headline_items.append(f"Estimated {int(round(total_population_affected)):,} people within affected census-tract footprints")
    if total_high_vulnerability_population > 0.0:
        headline_items.append(
            "Estimated "
            f"{int(round(total_high_vulnerability_population)):,} people in high social-vulnerability tracts within affected areas"
        )
    for row in rollup_rows:
        if row["metric"] == POINT_METRIC and int(row["affected_value"]) > 0 and row["category"] in {"healthcare", "emergency_response", "power"}:
            headline_items.append(f"{int(row['affected_value'])} {row['display_name'].lower()} affected")

    return {
        "rows": rollup_rows,
        "affected_site_count_total": int(total_sites),
        "affected_road_length_m": round(float(total_road_length_m), 1),
        "population_affected_estimate": int(round(total_population_affected)),
        "high_vulnerability_population_affected_estimate": int(round(total_high_vulnerability_population)),
        "categories_impacted": int(sum(1 for row in rollup_rows if float(row["affected_value"]) > 0.0)),
        "headline_items": headline_items[:4],
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
                row = compute_exposure(
                    scenario["geojson"],
                    layer_path,
                    layer_name=layer_name,
                    value_field=layer_cfg.get("value_field"),
                    weight_by_area=bool(layer_cfg.get("weight_by_area")),
                )
                exposure_rows.append(_enrich_exposure_row(row, layer_cfg))
            except Exception:
                continue

        rollup = _rollup_exposure_rows(exposure_rows)

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
            "affected_site_count_total": rollup["affected_site_count_total"],
            "affected_road_length_m": rollup["affected_road_length_m"],
            "population_affected_estimate": rollup["population_affected_estimate"],
            "high_vulnerability_population_affected_estimate": rollup["high_vulnerability_population_affected_estimate"],
            "categories_impacted": rollup["categories_impacted"],
            "impact_headline_items": rollup["headline_items"],
            "exposure_rollup": rollup["rows"],
            "exposure_summary": exposure_rows,
            "hotspots_geojson": hotspot_geojson,
            "hotspots_csv": hotspot_csv,
        }

    return summaries

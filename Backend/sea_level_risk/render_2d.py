from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pydeck as pdk


SCENARIO_2D_STYLE = {
    "plus_20cm": {"rgba": [69, 179, 224, 90], "label": "+20cm"},
    "plus_50cm": {"rgba": [30, 136, 229, 130], "label": "+50cm"},
    "plus_100cm": {"rgba": [13, 71, 161, 170], "label": "+100cm"},
}


def _load_geojson_features(path: str, scenario: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []

    gdf = gpd.read_file(p)
    if gdf.empty:
        return []
    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    style = SCENARIO_2D_STYLE.get(scenario, {"rgba": [65, 105, 225, 120], "label": scenario})
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {k: v for k, v in row.drop(labels=["geometry"]).to_dict().items() if pd.notna(v)}
        props.update(
            {
                "scenario": scenario,
                "scenario_label": style["label"],
                "fill_r": style["rgba"][0],
                "fill_g": style["rgba"][1],
                "fill_b": style["rgba"][2],
                "fill_a": style["rgba"][3],
            }
        )
        features.append({"type": "Feature", "geometry": geom.__geo_interface__, "properties": props})
    return features


def _load_hotspot_points(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []

    gdf = gpd.read_file(p)
    if gdf.empty:
        return []
    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type not in {"Point", "MultiPoint"}:
            continue
        points = [geom] if geom.geom_type == "Point" else list(geom.geoms)
        for pt in points:
            rows.append(
                {
                    "lon": float(pt.x),
                    "lat": float(pt.y),
                    "priority_score": float(row.get("priority_score", 0.0)),
                    "risk_level": row.get("risk_level", "unknown"),
                    "scenario": row.get("scenario", "unknown"),
                }
            )
    return rows


def build_2d_layers(
    scenario_items: list[dict],
    hotspot_geojson: str | None = None,
) -> tuple[list[pdk.Layer], pdk.ViewState | None]:
    all_features: list[dict] = []
    all_bounds = []
    for item in scenario_items:
        scenario = item.get("scenario", "unknown")
        geojson = item.get("flood_geojson") or item.get("geojson")
        features = _load_geojson_features(geojson, scenario)
        if not features:
            continue
        all_features.extend(features)

        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        minx, miny, maxx, maxy = gdf.total_bounds
        all_bounds.append((minx, miny, maxx, maxy))

    if not all_features:
        return [], None

    merged_geojson = {"type": "FeatureCollection", "features": all_features}
    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        data=merged_geojson,
        pickable=True,
        stroked=False,
        filled=True,
        extruded=False,
        get_fill_color="[properties.fill_r, properties.fill_g, properties.fill_b, properties.fill_a]",
    )

    layers: list[pdk.Layer] = [polygon_layer]

    hotspot_rows = _load_hotspot_points(hotspot_geojson) if hotspot_geojson else []
    if hotspot_rows:
        hotspot_layer = pdk.Layer(
            "ScatterplotLayer",
            data=hotspot_rows,
            get_position="[lon, lat]",
            get_radius=35,
            get_fill_color=[255, 82, 82, 220],
            pickable=True,
        )
        layers.append(hotspot_layer)
        hotspot_df = pd.DataFrame(hotspot_rows)
        all_bounds.append((hotspot_df["lon"].min(), hotspot_df["lat"].min(), hotspot_df["lon"].max(), hotspot_df["lat"].max()))

    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)
    center_lon = (minx + maxx) / 2.0
    center_lat = (miny + maxy) / 2.0
    span = max(maxx - minx, maxy - miny)
    zoom = 8
    if span < 0.2:
        zoom = 11
    elif span < 0.5:
        zoom = 10
    elif span < 1.0:
        zoom = 9

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0)
    return layers, view_state


def export_2d_feature_collection(scenario_items: list[dict], out_geojson: str) -> str:
    all_features: list[dict] = []
    for item in scenario_items:
        all_features.extend(_load_geojson_features(item.get("flood_geojson") or item.get("geojson"), item.get("scenario", "unknown")))
    out_path = Path(out_geojson)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": all_features}), encoding="utf-8")
    return str(out_path)

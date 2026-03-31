import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


EQUAL_AREA_EPSG = 6933
SCENARIO_WEIGHT = {"plus_20cm": 0.4, "plus_50cm": 0.7, "plus_100cm": 1.0}


def _risk_label(score: float) -> str:
    if score < 25:
        return "low"
    if score < 50:
        return "moderate"
    if score < 75:
        return "high"
    return "critical"


def _rank_hotspots_dataframe(scenario_records: list[dict]) -> gpd.GeoDataFrame:
    rows = []

    for rec in scenario_records:
        scenario = rec.get("scenario")
        geojson_path = rec.get("geojson")
        flood_ratio = float(rec.get("flood_ratio", 0.0))

        if not geojson_path or not Path(geojson_path).exists():
            continue

        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            continue

        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        if gdf.empty:
            continue

        gdf_area = gdf.to_crs(epsg=EQUAL_AREA_EPSG)
        areas = gdf_area.geometry.area.to_numpy(dtype=float)
        reps = gdf_area.geometry.representative_point()
        reps_ll = gpd.GeoSeries(reps, crs=f"EPSG:{EQUAL_AREA_EPSG}").to_crs(epsg=4326)

        for idx, (geom, area_m2, rep_pt) in enumerate(zip(gdf.geometry, areas, reps_ll), start=1):
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_weight": SCENARIO_WEIGHT.get(scenario, 0.5),
                    "polygon_id": idx,
                    "flood_ratio": flood_ratio,
                    "area_m2": float(area_m2),
                    "centroid_lon": float(rep_pt.x),
                    "centroid_lat": float(rep_pt.y),
                    "geometry": geom,
                }
            )

    if not rows:
        raise RuntimeError("No scenario polygons available to build hotspots.")

    df = pd.DataFrame(rows)
    area_norm = (df["area_m2"] - df["area_m2"].min()) / (df["area_m2"].max() - df["area_m2"].min() + 1e-9)

    # Priority score (0-100): scenario severity + flood ratio + polygon area
    df["priority_score"] = (
        100.0
        * (
            0.45 * df["scenario_weight"]
            + 0.40 * df["flood_ratio"].clip(0.0, 1.0)
            + 0.15 * area_norm.clip(0.0, 1.0)
        )
    )

    df = df.sort_values(["scenario", "priority_score"], ascending=[True, False]).reset_index(drop=True)
    df["rank"] = df.groupby("scenario").cumcount() + 1
    df["risk_level"] = df["priority_score"].apply(_risk_label)
    df["hotspot_label"] = df.apply(lambda row: f"{row['scenario']} hotspot #{int(row['rank'])}", axis=1)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def rank_hotspots_from_scenarios(
    scenario_records: list[dict],
    top_n: int = 20,
) -> gpd.GeoDataFrame:
    gdf = _rank_hotspots_dataframe(scenario_records)
    top = (
        gdf.sort_values(["scenario", "priority_score"], ascending=[True, False])
        .groupby("scenario", group_keys=False)
        .head(top_n)
        .copy()
    )
    return top.sort_values("priority_score", ascending=False).reset_index(drop=True)


def hotspot_records_from_gdf(hotspot_gdf: gpd.GeoDataFrame, limit: int = 10, scenario: str | None = None) -> list[dict]:
    if hotspot_gdf.empty:
        return []

    filtered = hotspot_gdf
    if scenario:
        filtered = filtered[filtered["scenario"] == scenario].copy()
    if filtered.empty:
        return []

    filtered = filtered.sort_values("priority_score", ascending=False).head(limit).copy()
    return filtered.drop(columns=["geometry"]).to_dict(orient="records")


def ensure_hotspots_from_scenarios(
    scenario_records: list[dict],
    out_geojson: str,
    out_csv: str,
    top_n: int = 20,
) -> gpd.GeoDataFrame:
    out_geo = Path(out_geojson)
    out_csv_path = Path(out_csv)
    source_paths = [Path(rec["geojson"]) for rec in scenario_records if rec.get("geojson") and Path(rec["geojson"]).exists()]
    latest_source_mtime = max((path.stat().st_mtime for path in source_paths), default=0.0)

    if out_geo.exists() and out_csv_path.exists():
        hotspot_mtime = min(out_geo.stat().st_mtime, out_csv_path.stat().st_mtime)
        if hotspot_mtime >= latest_source_mtime:
            cached = gpd.read_file(out_geo)
            expected_scenarios = {rec.get("scenario") for rec in scenario_records if rec.get("scenario")}
            cached_scenarios = set(cached["scenario"].unique()) if not cached.empty and "scenario" in cached.columns else set()
            required_columns = {"scenario", "rank", "priority_score", "centroid_lon", "centroid_lat", "area_m2"}
            if (
                not cached.empty
                and expected_scenarios.issubset(cached_scenarios)
                and required_columns.issubset(set(cached.columns))
            ):
                return cached

    hotspot_gdf = rank_hotspots_from_scenarios(scenario_records, top_n=top_n)
    out_geo.parent.mkdir(parents=True, exist_ok=True)
    hotspot_gdf.to_file(out_geo, driver="GeoJSON")
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    hotspot_gdf.drop(columns=["geometry"]).to_csv(out_csv_path, index=False)
    return hotspot_gdf


def build_hotspots_from_scenarios(
    scenario_records: list[dict],
    out_geojson: str,
    out_csv: str,
    top_n: int = 20,
) -> dict:
    gdf_out = ensure_hotspots_from_scenarios(
        scenario_records=scenario_records,
        out_geojson=out_geojson,
        out_csv=out_csv,
        top_n=top_n,
    )
    out_geo = Path(out_geojson)
    out_table = Path(out_csv)

    return {
        "count": int(len(gdf_out)),
        "out_geojson": str(out_geo),
        "out_csv": str(out_table),
    }


def main():
    parser = argparse.ArgumentParser(description="Build anti-flood hotspots from scenario polygons")
    parser.add_argument("--scenario-dir", required=True, help="Folder containing flood_plus_*.geojson")
    parser.add_argument("--out-geojson", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    sdir = Path(args.scenario_dir)
    records = []
    for s in ["plus_20cm", "plus_50cm", "plus_100cm"]:
        p = sdir / f"flood_{s}.geojson"
        if p.exists():
            records.append({"scenario": s, "geojson": str(p), "flood_ratio": 0.0})

    result = build_hotspots_from_scenarios(records, args.out_geojson, args.out_csv, top_n=args.top)
    print(result)


if __name__ == "__main__":
    main()

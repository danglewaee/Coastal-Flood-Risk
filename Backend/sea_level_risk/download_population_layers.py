from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

from .city_registry import load_city_registry


ACS_YEAR = 2023
ACS_POP_VARIABLE = "B01003_001E"
TIGER_TRACT_URL = "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_{state_fips}_tract.zip"
ACS_TRACT_URL = "https://api.census.gov/data/{year}/acs/acs5"
DEFAULT_RADIUS_KM = 24.0

CITY_POPULATION_STATE_FIPS = {
    "boston": ["25"],
    "newyork": ["34", "36"],
    "miami": ["12"],
    "sanfrancisco": ["06"],
}


def _bbox_for_city(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    lat_delta = radius_km / 111.32
    lon_delta = radius_km / max(111.32 * math.cos(math.radians(lat)), 1e-6)
    south = lat - lat_delta
    north = lat + lat_delta
    west = lon - lon_delta
    east = lon + lon_delta
    return south, west, north, east


def _download_state_tracts(state_fips: str) -> gpd.GeoDataFrame:
    url = TIGER_TRACT_URL.format(state_fips=state_fips)
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / f"tl_2023_{state_fips}_tract.zip"
        response = requests.get(url, timeout=240)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
        gdf = gpd.read_file(f"zip://{zip_path}")
    return gdf.to_crs("EPSG:4326")


def _fetch_county_population(state_fips: str, county_fips: str) -> pd.DataFrame:
    response = requests.get(
        ACS_TRACT_URL.format(year=ACS_YEAR),
        params={
            "get": f"NAME,{ACS_POP_VARIABLE}",
            "for": "tract:*",
            "in": f"state:{state_fips} county:{county_fips}",
        },
        timeout=240,
    )
    response.raise_for_status()
    data = response.json()
    header, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=header)
    df["population_total"] = pd.to_numeric(df[ACS_POP_VARIABLE], errors="coerce").fillna(0.0)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    return df[["GEOID", "population_total", "NAME"]]


def download_city_population_layer(
    city_key: str,
    *,
    out_root: str = "data/exposure",
    radius_km: float | None = None,
) -> dict:
    registry = load_city_registry()
    cfg = registry.get(city_key)
    if cfg is None:
        raise ValueError(f"Unknown city '{city_key}'.")

    lat = cfg.get("lat")
    lon = cfg.get("lon")
    if lat is None or lon is None:
        raise ValueError(f"City '{city_key}' is missing lat/lon in city_registry.json.")

    state_fips_list = CITY_POPULATION_STATE_FIPS.get(city_key)
    if not state_fips_list:
        raise ValueError(f"Population download is not configured for city '{city_key}'.")

    effective_radius = float(radius_km or DEFAULT_RADIUS_KM)
    south, west, north, east = _bbox_for_city(float(lat), float(lon), effective_radius)
    city_bbox = box(west, south, east, north)

    tract_frames = []
    for state_fips in state_fips_list:
        state_gdf = _download_state_tracts(state_fips)
        clipped = state_gdf[state_gdf.intersects(city_bbox)].copy()
        if clipped.empty:
            continue

        county_frames = []
        for county_fips in sorted(clipped["COUNTYFP"].dropna().astype(str).unique().tolist()):
            county_frames.append(_fetch_county_population(state_fips, county_fips.zfill(3)))
        if county_frames:
            acs_df = pd.concat(county_frames, ignore_index=True).drop_duplicates(subset=["GEOID"])
            clipped["GEOID"] = clipped["GEOID"].astype(str)
            clipped = clipped.merge(acs_df, on="GEOID", how="left")
            clipped["population_total"] = pd.to_numeric(clipped["population_total"], errors="coerce").fillna(0.0)
            tract_frames.append(clipped)

    if not tract_frames:
        raise RuntimeError(f"No census tracts found for {city_key} within the configured search radius.")

    population_gdf = pd.concat(tract_frames, ignore_index=True)
    population_gdf = gpd.GeoDataFrame(population_gdf, geometry="geometry", crs="EPSG:4326")
    population_gdf = population_gdf[population_gdf.intersects(city_bbox)].copy()

    keep_cols = [
        "GEOID",
        "NAME",
        "population_total",
        "STATEFP",
        "COUNTYFP",
        "TRACTCE",
        "geometry",
    ]
    keep_cols = [col for col in keep_cols if col in population_gdf.columns]
    population_gdf = population_gdf[keep_cols]

    city_dir = Path(out_root) / city_key
    city_dir.mkdir(parents=True, exist_ok=True)
    out_path = city_dir / "population_tracts.geojson"
    population_gdf.to_file(out_path, driver="GeoJSON")

    return {
        "city": city_key,
        "display_name": cfg.get("display_name", city_key),
        "path": str(out_path),
        "features": int(len(population_gdf)),
        "population_total": int(round(float(population_gdf["population_total"].sum()))),
        "states": state_fips_list,
        "bbox": {"south": south, "west": west, "north": north, "east": east},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Census tract population layers for supported cities.")
    parser.add_argument("--cities", nargs="*", help="City keys from city_registry.json.")
    parser.add_argument("--all-supported", action="store_true", help="Download all cities with configured Census population support.")
    parser.add_argument("--out-root", default="data/exposure", help="Output root directory for GeoJSON population layers.")
    parser.add_argument("--radius-km", type=float, help="Override search radius in kilometers for all selected cities.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_supported:
        cities = list(CITY_POPULATION_STATE_FIPS.keys())
    else:
        cities = [item.strip().lower() for item in (args.cities or []) if item.strip()]
    if not cities:
        raise SystemExit("Provide --cities <city...> or use --all-supported.")

    results = []
    for city_key in cities:
        results.append(download_city_population_layer(city_key, out_root=args.out_root, radius_km=args.radius_km))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

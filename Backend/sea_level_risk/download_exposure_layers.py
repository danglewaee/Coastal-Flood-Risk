from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import LineString, Point

from .city_registry import load_city_registry


OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
DEFAULT_RADIUS_KM = 20.0
CITY_RADIUS_KM = {
    "honolulu": 18.0,
    "boston": 24.0,
    "newyork": 24.0,
    "miami": 24.0,
    "sanfrancisco": 22.0,
    "jakarta": 20.0,
    "amsterdam": 20.0,
}


def _bbox_for_city(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    lat_delta = radius_km / 111.32
    lon_delta = radius_km / max(111.32 * math.cos(math.radians(lat)), 1e-6)
    south = lat - lat_delta
    north = lat + lat_delta
    west = lon - lon_delta
    east = lon + lon_delta
    return south, west, north, east


def _roads_query(south: float, west: float, north: float, east: float) -> str:
    return f"""
[out:json][timeout:180];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential"]({south},{west},{north},{east});
);
out tags geom;
""".strip()


def _facilities_query(south: float, west: float, north: float, east: float) -> str:
    return f"""
[out:json][timeout:180];
(
  nwr["amenity"~"hospital|clinic|police|fire_station"]({south},{west},{north},{east});
  nwr["emergency"="fire_station"]({south},{west},{north},{east});
);
out center tags;
""".strip()


def _post_query(query: str, endpoint: str = OVERPASS_ENDPOINT, fallback_endpoints: list[str] | None = None) -> dict:
    endpoints = [endpoint] + [item for item in (fallback_endpoints or OVERPASS_ENDPOINTS) if item != endpoint]
    last_error: Exception | None = None
    for idx, current in enumerate(endpoints):
        try:
            response = requests.post(current, data={"data": query}, timeout=240)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if idx < len(endpoints) - 1:
                time.sleep(2.0)
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("No Overpass endpoint could be reached.")


def _roads_gdf(overpass_json: dict) -> gpd.GeoDataFrame:
    features = []
    for element in overpass_json.get("elements", []):
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        coords = [(float(item["lon"]), float(item["lat"])) for item in geometry]
        features.append(
            {
                "osm_id": f"{element.get('type', 'way')}/{element.get('id')}",
                "name": (element.get("tags") or {}).get("name"),
                "highway": (element.get("tags") or {}).get("highway"),
                "bridge": (element.get("tags") or {}).get("bridge"),
                "geometry": LineString(coords),
            }
        )

    return gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")


def _facility_point(element: dict) -> Point | None:
    center = element.get("center")
    if center:
        return Point(float(center["lon"]), float(center["lat"]))
    if "lon" in element and "lat" in element:
        return Point(float(element["lon"]), float(element["lat"]))
    return None


def _facilities_gdf(overpass_json: dict) -> gpd.GeoDataFrame:
    features = []
    for element in overpass_json.get("elements", []):
        geom = _facility_point(element)
        if geom is None:
            continue
        tags = element.get("tags") or {}
        features.append(
            {
                "osm_id": f"{element.get('type', 'nwr')}/{element.get('id')}",
                "name": tags.get("name"),
                "amenity": tags.get("amenity"),
                "emergency": tags.get("emergency"),
                "operator": tags.get("operator"),
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")


def download_city_exposure_layers(
    city_key: str,
    *,
    out_root: str = "data/exposure",
    radius_km: float | None = None,
    endpoint: str = OVERPASS_ENDPOINT,
    skip_existing: bool = False,
) -> dict:
    registry = load_city_registry()
    cfg = registry.get(city_key)
    if cfg is None:
        raise ValueError(f"Unknown city '{city_key}'.")

    lat = cfg.get("lat")
    lon = cfg.get("lon")
    if lat is None or lon is None:
        raise ValueError(f"City '{city_key}' is missing lat/lon in city_registry.json.")

    effective_radius = float(radius_km or CITY_RADIUS_KM.get(city_key, DEFAULT_RADIUS_KM))
    south, west, north, east = _bbox_for_city(float(lat), float(lon), effective_radius)

    city_dir = Path(out_root) / city_key
    city_dir.mkdir(parents=True, exist_ok=True)

    roads_path = city_dir / "roads.geojson"
    facilities_path = city_dir / "critical_facilities.geojson"

    results = {
        "city": city_key,
        "display_name": cfg.get("display_name", city_key),
        "bbox": {"south": south, "west": west, "north": north, "east": east},
        "radius_km": effective_radius,
        "roads_path": str(roads_path),
        "critical_facilities_path": str(facilities_path),
    }

    if not (skip_existing and roads_path.exists()):
        roads = _roads_gdf(_post_query(_roads_query(south, west, north, east), endpoint=endpoint))
        roads.to_file(roads_path, driver="GeoJSON")
        results["roads_features"] = int(len(roads))
    else:
        results["roads_features"] = int(len(gpd.read_file(roads_path)))

    if not (skip_existing and facilities_path.exists()):
        facilities = _facilities_gdf(_post_query(_facilities_query(south, west, north, east), endpoint=endpoint))
        facilities.to_file(facilities_path, driver="GeoJSON")
        results["critical_facilities_features"] = int(len(facilities))
    else:
        results["critical_facilities_features"] = int(len(gpd.read_file(facilities_path)))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download real exposure layers from OSM Overpass for supported cities.")
    parser.add_argument("--cities", nargs="*", help="City keys from city_registry.json.")
    parser.add_argument("--all-known", action="store_true", help="Download exposure layers for all known city registry entries.")
    parser.add_argument("--out-root", default="data/exposure", help="Output root directory for GeoJSON exposure layers.")
    parser.add_argument("--radius-km", type=float, help="Override search radius in kilometers for all selected cities.")
    parser.add_argument("--endpoint", default=OVERPASS_ENDPOINT, help="Overpass API endpoint.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip downloads when target GeoJSON already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_city_registry()
    if args.all_known:
        cities = list(registry.keys())
    else:
        cities = [item.strip().lower() for item in (args.cities or []) if item.strip()]

    if not cities:
        raise SystemExit("Provide --cities <city...> or use --all-known.")

    results = []
    for city_key in cities:
        results.append(
            download_city_exposure_layers(
                city_key,
                out_root=args.out_root,
                radius_km=args.radius_km,
                endpoint=args.endpoint,
                skip_existing=args.skip_existing,
            )
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

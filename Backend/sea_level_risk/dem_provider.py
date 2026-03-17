import math
from pathlib import Path

import requests


NOAA_STATION_META = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}.json"
COPERNICUS_TEMPLATE = "https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif"


def get_station_lat_lon_noaa(station: str) -> tuple[float, float]:
    url = NOAA_STATION_META.format(station=station)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()

    stations = payload.get("stations", [])
    if not stations:
        raise RuntimeError(f"No station metadata found for {station}")

    lat = float(stations[0]["lat"])
    lon = float(stations[0]["lng"])
    return lat, lon


def copernicus_tile_name(lat: float, lon: float) -> str:
    lat_i = math.floor(lat)
    lon_i = math.floor(lon)

    lat_tag = f"{'N' if lat_i >= 0 else 'S'}{abs(lat_i):02d}_00"
    lon_tag = f"{'E' if lon_i >= 0 else 'W'}{abs(lon_i):03d}_00"
    return f"Copernicus_DSM_COG_10_{lat_tag}_{lon_tag}_DEM"


def ensure_dem_for_lat_lon(lat: float, lon: float, cache_dir: str = "data/dem_cache") -> str:
    tile = copernicus_tile_name(lat, lon)

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tile}.tif"
    if out_path.exists():
        return str(out_path)

    url = COPERNICUS_TEMPLATE.format(tile=tile)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return str(out_path)


def ensure_dem_for_station(station: str, cache_dir: str = "data/dem_cache") -> str:
    lat, lon = get_station_lat_lon_noaa(station)
    return ensure_dem_for_lat_lon(lat, lon, cache_dir=cache_dir)

import math
from pathlib import Path

import rasterio
import requests
from rasterio.windows import from_bounds


NOAA_STATION_META = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}.json"
COPERNICUS_TEMPLATE = "https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif"

DEFAULT_CITY_CLIP_RADIUS_KM = 20.0
CITY_DEM_CLIP_RADIUS_KM = {
    "honolulu": 18.0,
    "boston": 24.0,
    "newyork": 24.0,
    "jakarta": 20.0,
    "amsterdam": 20.0,
    "miami": 18.0,
    "sanfrancisco": 20.0,
}


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


def _bbox_for_lat_lon(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    lat_delta = radius_km / 111.32
    lon_delta = radius_km / max(111.32 * math.cos(math.radians(lat)), 1e-6)
    south = lat - lat_delta
    north = lat + lat_delta
    west = lon - lon_delta
    east = lon + lon_delta
    return south, west, north, east


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


def cached_dem_for_lat_lon(lat: float, lon: float, cache_dir: str = "data/dem_cache") -> str | None:
    tile = copernicus_tile_name(lat, lon)
    out_path = Path(cache_dir) / f"{tile}.tif"
    return str(out_path) if out_path.exists() else None


def cached_city_dem(city_key: str, cache_dir: str = "data/dem_city_cache") -> str | None:
    out_path = Path(cache_dir) / f"{city_key}_dem_clip.tif"
    return str(out_path) if out_path.exists() else None


def clip_dem_for_city(
    city_key: str,
    lat: float,
    lon: float,
    source_dem_path: str,
    *,
    cache_dir: str = "data/dem_city_cache",
    radius_km: float | None = None,
) -> str:
    src_path = Path(source_dem_path)
    if not src_path.exists():
        raise FileNotFoundError(f"DEM source not found: {source_dem_path}")

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{city_key}_dem_clip.tif"

    if out_path.exists() and out_path.stat().st_mtime >= src_path.stat().st_mtime:
        return str(out_path)

    effective_radius = float(radius_km or CITY_DEM_CLIP_RADIUS_KM.get(city_key, DEFAULT_CITY_CLIP_RADIUS_KM))
    south, west, north, east = _bbox_for_lat_lon(lat, lon, effective_radius)

    with rasterio.open(src_path) as src:
        clip_west = max(west, src.bounds.left)
        clip_south = max(south, src.bounds.bottom)
        clip_east = min(east, src.bounds.right)
        clip_north = min(north, src.bounds.top)
        if clip_west >= clip_east or clip_south >= clip_north:
            raise RuntimeError(f"Requested clip bbox for {city_key} falls outside DEM bounds.")

        window = from_bounds(clip_west, clip_south, clip_east, clip_north, src.transform)
        window = window.round_offsets().round_lengths()
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            {
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": transform,
            }
        )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    return str(out_path)


def ensure_city_dem_for_lat_lon(
    city_key: str,
    lat: float,
    lon: float,
    *,
    raw_cache_dir: str = "data/dem_cache",
    city_cache_dir: str = "data/dem_city_cache",
    radius_km: float | None = None,
) -> str:
    source_dem = ensure_dem_for_lat_lon(lat, lon, cache_dir=raw_cache_dir)
    return clip_dem_for_city(
        city_key=city_key,
        lat=lat,
        lon=lon,
        source_dem_path=source_dem,
        cache_dir=city_cache_dir,
        radius_km=radius_km,
    )


def city_dem_for_lat_lon_if_cached(
    city_key: str,
    lat: float,
    lon: float,
    *,
    raw_cache_dir: str = "data/dem_cache",
    city_cache_dir: str = "data/dem_city_cache",
    radius_km: float | None = None,
) -> str | None:
    clipped = cached_city_dem(city_key, cache_dir=city_cache_dir)
    if clipped:
        return clipped

    raw = cached_dem_for_lat_lon(lat, lon, cache_dir=raw_cache_dir)
    if not raw:
        return None

    return clip_dem_for_city(
        city_key=city_key,
        lat=lat,
        lon=lon,
        source_dem_path=raw,
        cache_dir=city_cache_dir,
        radius_km=radius_km,
    )


def ensure_dem_for_station(station: str, cache_dir: str = "data/dem_cache") -> str:
    lat, lon = get_station_lat_lon_noaa(station)
    return ensure_dem_for_lat_lon(lat, lon, cache_dir=cache_dir)

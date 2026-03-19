from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


DEFAULT_CITY_REGISTRY = {
    "honolulu": {
        "display_name": "Honolulu, HI",
        "provider": "noaa",
        "provider_label": "NOAA CO-OPS",
        "station_id": "1612340",
        "provider_product": "water_level",
        "value_column": "sea_level",
        "dem_path": "data/honolulu_dem.tif",
        "admin_boundary": None,
        "timezone": "Pacific/Honolulu",
        "lat": 21.3069,
        "lon": -157.8583,
        "support_tier": "official_realtime",
        "proxy_mode": "direct_station",
        "forecast_mode": "model_recursive",
        "notes": "Direct NOAA gauge with local trained deep-learning model.",
    },
    "boston": {
        "display_name": "Boston Harbor, MA",
        "provider": "noaa",
        "provider_label": "NOAA CO-OPS",
        "station_id": "8443970",
        "provider_product": "water_level",
        "value_column": "sea_level",
        "dem_path": None,
        "admin_boundary": None,
        "timezone": "America/New_York",
        "lat": 42.3601,
        "lon": -71.0589,
        "support_tier": "official_realtime",
        "proxy_mode": "direct_station",
        "forecast_mode": "city_model_or_baseline",
        "notes": "Direct NOAA station. Uses a city-specific deep-learning model when available; otherwise falls back to a tide-aware baseline.",
    },
    "newyork": {
        "display_name": "The Battery, New York",
        "provider": "noaa",
        "provider_label": "NOAA CO-OPS",
        "station_id": "8518750",
        "provider_product": "water_level",
        "value_column": "sea_level",
        "dem_path": None,
        "admin_boundary": None,
        "timezone": "America/New_York",
        "lat": 40.7128,
        "lon": -74.0060,
        "support_tier": "official_realtime",
        "proxy_mode": "direct_station",
        "forecast_mode": "city_model_or_baseline",
        "notes": "Direct NOAA station. Uses a city-specific deep-learning model when available; otherwise falls back to a tide-aware baseline.",
    },
    "jakarta": {
        "display_name": "Kolinamil, Jakarta Port",
        "provider": "ioc",
        "provider_label": "UNESCO IOC Sea Level Monitoring",
        "station_code": "koli",
        "value_column": "prs",
        "dem_path": None,
        "admin_boundary": None,
        "timezone": "Asia/Jakarta",
        "lat": -6.2088,
        "lon": 106.8456,
        "support_tier": "experimental_realtime",
        "proxy_mode": "direct_station",
        "forecast_mode": "city_model_or_baseline",
        "notes": "Experimental direct IOC feed. Uses a city-specific deep-learning model when an hourly training set exists; otherwise falls back to a tide-aware baseline.",
    },
    "amsterdam": {
        "display_name": "Amsterdam-region proxy (Hoek van Holland)",
        "provider": "ioc",
        "provider_label": "UNESCO IOC Sea Level Monitoring",
        "station_code": "hoek",
        "value_column": "flt",
        "dem_path": None,
        "admin_boundary": None,
        "timezone": "Europe/Amsterdam",
        "lat": 52.3676,
        "lon": 4.9041,
        "support_tier": "proxy_delayed",
        "proxy_mode": "regional_proxy",
        "forecast_mode": "city_model_or_baseline",
        "notes": "Nearest feasible open-coast proxy for Amsterdam. Uses a city-specific deep-learning model when an hourly training set exists; otherwise falls back to a tide-aware baseline. Treat outputs as regional proxy, not a true realtime Amsterdam city gauge.",
    },
    "miami": {
        "display_name": "Virginia Key, Miami",
        "provider": "noaa",
        "provider_label": "NOAA CO-OPS",
        "station_id": "8723214",
        "provider_product": "water_level",
        "value_column": "sea_level",
        "dem_path": None,
        "admin_boundary": None,
        "timezone": "America/New_York",
        "lat": 25.7617,
        "lon": -80.1918,
        "support_tier": "official_realtime",
        "proxy_mode": "direct_station",
        "forecast_mode": "city_model_or_baseline",
        "notes": "Direct NOAA station. Uses a city-specific deep-learning model when available; otherwise falls back to a tide-aware baseline.",
    },
    "sanfrancisco": {
        "display_name": "San Francisco",
        "provider": "noaa",
        "provider_label": "NOAA CO-OPS",
        "station_id": "9414290",
        "provider_product": "water_level",
        "value_column": "sea_level",
        "dem_path": None,
        "admin_boundary": None,
        "timezone": "America/Los_Angeles",
        "lat": 37.7749,
        "lon": -122.4194,
        "support_tier": "official_realtime",
        "proxy_mode": "direct_station",
        "forecast_mode": "city_model_or_baseline",
        "notes": "Direct NOAA station. Uses a city-specific deep-learning model when available; otherwise falls back to a tide-aware baseline.",
    },
}


def _normalize_city_entry(city_key: str, raw: dict) -> dict:
    base = deepcopy(DEFAULT_CITY_REGISTRY.get(city_key, {}))
    merged = {**base, **raw}

    if "station" in merged and "station_id" not in merged:
        merged["station_id"] = merged["station"]
    merged.pop("station", None)

    if not merged.get("provider"):
        merged["provider"] = "noaa" if merged.get("station_id") else "ioc"
    if not merged.get("provider_label"):
        merged["provider_label"] = "NOAA CO-OPS" if merged["provider"] == "noaa" else "UNESCO IOC Sea Level Monitoring"
    if merged["provider"] == "noaa" and not merged.get("provider_product"):
        merged["provider_product"] = "water_level"
    if not merged.get("forecast_mode"):
        merged["forecast_mode"] = "city_model_or_baseline"
    if not merged.get("support_tier"):
        merged["support_tier"] = "official_realtime" if merged["provider"] == "noaa" else "experimental_realtime"
    if not merged.get("proxy_mode"):
        merged["proxy_mode"] = "direct_station"
    if not merged.get("value_column"):
        merged["value_column"] = "sea_level" if merged["provider"] == "noaa" else None

    return merged


def load_city_registry(path: str = "Backend/sea_level_risk/city_registry.json") -> dict:
    p = Path(path)
    if p.exists():
        loaded = json.loads(p.read_text(encoding="utf-8-sig"))
    else:
        loaded = deepcopy(DEFAULT_CITY_REGISTRY)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(loaded, indent=2), encoding="utf-8")

    normalized = {key: _normalize_city_entry(key, value) for key, value in loaded.items()}

    for key, value in DEFAULT_CITY_REGISTRY.items():
        if key not in normalized:
            normalized[key] = deepcopy(value)

    return normalized

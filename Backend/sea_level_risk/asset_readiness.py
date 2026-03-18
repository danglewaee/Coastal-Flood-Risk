from __future__ import annotations

from pathlib import Path


REQUIRED_SCENARIOS = ("plus_20cm", "plus_50cm", "plus_100cm")


def city_map_status(city_key: str, outputs_root: str | Path = "Backend/sea_level_risk/outputs/realtime") -> dict:
    city_dir = Path(outputs_root) / city_key
    available = [name for name in REQUIRED_SCENARIOS if (city_dir / f"flood_{name}.geojson").exists()]
    missing = [name for name in REQUIRED_SCENARIOS if name not in available]
    return {
        "city": city_key,
        "city_dir": str(city_dir),
        "available_scenarios": available,
        "missing_scenarios": missing,
        "is_full_map_ready": len(available) == len(REQUIRED_SCENARIOS),
        "is_partial_map_ready": len(available) > 0 and len(available) < len(REQUIRED_SCENARIOS),
        "is_forecast_only": len(available) == 0,
    }


def split_city_keys_by_map_status(city_keys: list[str], outputs_root: str | Path = "Backend/sea_level_risk/outputs/realtime") -> tuple[list[str], list[str], list[str]]:
    full: list[str] = []
    partial: list[str] = []
    forecast_only: list[str] = []

    for city_key in city_keys:
        status = city_map_status(city_key, outputs_root=outputs_root)
        if status["is_full_map_ready"]:
            full.append(city_key)
        elif status["is_partial_map_ready"]:
            partial.append(city_key)
        else:
            forecast_only.append(city_key)

    return full, partial, forecast_only

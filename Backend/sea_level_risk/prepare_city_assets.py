from __future__ import annotations

import argparse
from pathlib import Path

from .city_registry import load_city_registry
from .qgis.prepare_qgis_package import prepare_package
from .realtime_api import RealtimeService


def prepare_assets(
    cities: list[str],
    model_path: str,
    metadata_path: str,
    horizon: int,
    hours_back: int,
    auto_dem: bool,
    qgis_out_root: str,
) -> list[dict]:
    service = RealtimeService(model_path=model_path, metadata_path=metadata_path, default_dem_path=None)
    registry = load_city_registry()
    results: list[dict] = []

    for city in cities:
        city_key = city.strip().lower()
        if city_key not in registry:
            raise ValueError(f"Unknown city '{city_key}'. Available: {list(registry.keys())}")

        payload = service.predict(
            city=city_key,
            station=None,
            provider=None,
            horizon=horizon,
            hours_back=hours_back,
            datum="MSL",
            dem_path=None,
            auto_dem=auto_dem,
        )

        package_dir = None
        if payload.get("dem_path") and payload.get("scenarios"):
            rt_dir = Path("Backend/sea_level_risk/outputs/realtime") / city_key
            package_dir = prepare_package(
                city=city_key,
                dem_path=payload["dem_path"],
                realtime_dir=str(rt_dir),
                out_root=qgis_out_root,
            )

        results.append(
            {
                "city": city_key,
                "display_name": payload.get("display_name"),
                "provider": payload.get("provider"),
                "support_tier": payload.get("support_tier"),
                "station": payload.get("station"),
                "forecast_mode_used": payload.get("model", {}).get("forecast_mode_used"),
                "source_status": payload.get("source", {}).get("status"),
                "observation_delay_hours": payload.get("source", {}).get("observation_delay_hours"),
                "dem_path": payload.get("dem_path"),
                "scenario_count": len(payload.get("scenarios", [])),
                "qgis_package_dir": str(package_dir) if package_dir else None,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Warm up DEM/scenario/QGIS assets for selected cities.")
    parser.add_argument("--cities", default="boston,newyork", help="Comma-separated city keys from city_registry.json")
    parser.add_argument("--model", default="Backend/sea_level_risk/outputs/sea_level_axial_lstm.keras")
    parser.add_argument("--metadata", default="Backend/sea_level_risk/outputs/metadata.json")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--hours-back", type=int, default=96)
    parser.add_argument("--auto-dem", action="store_true", help="Fetch DEM automatically when missing")
    parser.add_argument("--qgis-out-root", default="Backend/sea_level_risk/outputs/qgis_packages")
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    results = prepare_assets(
        cities=cities,
        model_path=args.model,
        metadata_path=args.metadata,
        horizon=args.horizon,
        hours_back=args.hours_back,
        auto_dem=args.auto_dem,
        qgis_out_root=args.qgis_out_root,
    )
    print(results)


if __name__ == "__main__":
    main()

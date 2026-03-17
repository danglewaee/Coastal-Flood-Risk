from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model

from .city_registry import load_city_registry
from .data_providers import ensure_hours_back_coverage, fetch_ioc_recent, fetch_noaa_recent
from .dem_provider import ensure_dem_for_lat_lon, ensure_dem_for_station
from .forecast import recursive_forecast_with_loaded_model
from .forecast_baselines import tide_persistence_forecast_from_frame
from .gis import dem_to_flood_polygon
from .priority import build_hotspots_from_scenarios


DEFAULT_SCENARIOS_M = [0.2, 0.5, 1.0]


def _risk_label(flood_ratio: float) -> str:
    if flood_ratio < 0.001:
        return "low"
    if flood_ratio < 0.01:
        return "moderate"
    if flood_ratio < 0.05:
        return "high"
    return "critical"


def _slugify_station_ref(provider: str, station_ref: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in station_ref.lower())
    return f"{provider}_{safe}"


class RealtimeService:
    def __init__(self, model_path: str, metadata_path: str, default_dem_path: str | None = None):
        self.model = None
        self.metadata: dict = {}
        model_file = Path(model_path)
        metadata_file = Path(metadata_path)
        if model_file.exists() and metadata_file.exists():
            self.model = load_model(model_file, compile=False)
            self.metadata = json.loads(metadata_file.read_text(encoding="utf-8-sig"))

        self.default_dem_path = default_dem_path
        self.city_registry = load_city_registry()

    def _resolve_city_config(self, city: str | None, provider: str | None, station: str | None) -> tuple[str | None, dict]:
        if city:
            city_key = city.strip().lower()
            if city_key not in self.city_registry:
                raise ValueError(f"Unknown city '{city}'. Available: {list(self.city_registry.keys())}")
            return city_key, dict(self.city_registry[city_key])

        inferred_provider = (provider or "noaa").strip().lower()
        inferred_station = station or ("1612340" if inferred_provider == "noaa" else "koli")
        if inferred_provider not in {"noaa", "ioc"}:
            raise ValueError("provider must be one of: noaa, ioc")

        cfg = {
            "display_name": inferred_station,
            "provider": inferred_provider,
            "provider_label": "NOAA CO-OPS" if inferred_provider == "noaa" else "UNESCO IOC Sea Level Monitoring",
            "station_id": inferred_station if inferred_provider == "noaa" else None,
            "station_code": inferred_station if inferred_provider == "ioc" else None,
            "provider_product": "water_level" if inferred_provider == "noaa" else None,
            "value_column": "sea_level" if inferred_provider == "noaa" else None,
            "dem_path": self.default_dem_path,
            "admin_boundary": None,
            "timezone": "UTC",
            "lat": None,
            "lon": None,
            "support_tier": "ad_hoc_station",
            "proxy_mode": "direct_station",
            "forecast_mode": "tide_persistence",
            "notes": "Ad-hoc station request without a named city registry entry.",
        }
        return None, cfg

    def _resolve_dem_path(self, cfg: dict, dem_path: str | None, auto_dem: bool) -> str | None:
        resolved_dem = dem_path or cfg.get("dem_path")
        if resolved_dem and Path(resolved_dem).exists():
            return resolved_dem

        if not auto_dem:
            return resolved_dem if resolved_dem and Path(resolved_dem).exists() else None

        provider = cfg.get("provider")
        if provider == "noaa" and cfg.get("station_id"):
            return ensure_dem_for_station(cfg["station_id"])

        lat = cfg.get("lat")
        lon = cfg.get("lon")
        if lat is not None and lon is not None:
            return ensure_dem_for_lat_lon(float(lat), float(lon))

        return None

    def fetch_latest_series(
        self,
        cfg: dict,
        station_override: str | None,
        hours_back: int = 72,
        datum: str = "MSL",
    ) -> tuple[pd.DataFrame, dict]:
        provider = cfg.get("provider", "noaa")
        if provider == "noaa":
            station_ref = station_override or cfg.get("station_id")
            if not station_ref:
                raise ValueError("NOAA city config is missing station_id.")
            result = fetch_noaa_recent(
                station=station_ref,
                hours_back=hours_back,
                datum=datum,
                product=cfg.get("provider_product", "water_level"),
            )
        elif provider == "ioc":
            station_ref = station_override or cfg.get("station_code")
            if not station_ref:
                raise ValueError("IOC city config is missing station_code.")
            result = fetch_ioc_recent(
                station_code=station_ref,
                hours_back=hours_back,
                preferred_value_column=cfg.get("value_column"),
            )
        else:
            raise ValueError(f"Unsupported provider '{provider}'.")

        frame = ensure_hours_back_coverage(result.frame, hours_back=hours_back)
        meta = {
            "provider": result.provider,
            "provider_label": result.source_name,
            "station_ref": result.station_ref,
            "source_value_column": result.source_value_column,
            "observation_delay_hours": result.observation_delay_hours,
            "status": result.status,
            "note": result.note,
        }
        return frame, meta

    def _forecast(self, cfg: dict, df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, str, str | None]:
        requested_mode = cfg.get("forecast_mode", "tide_persistence")
        recent = df["sea_level"].to_numpy(dtype=np.float32)

        if requested_mode == "model_recursive" and self.model is not None and self.metadata:
            preds = recursive_forecast_with_loaded_model(
                model=self.model,
                metadata=self.metadata,
                recent_values=recent,
                horizon_hours=horizon,
            )
            return preds, "model_recursive", None

        note = None
        mode_used = requested_mode
        if requested_mode == "model_recursive":
            note = "Configured deep-learning model was unavailable for this request. Falling back to tide-aware baseline."
            mode_used = "tide_persistence_fallback"
        preds = tide_persistence_forecast_from_frame(df, horizon_hours=horizon)
        return preds, mode_used, note

    def predict(
        self,
        station: str | None,
        horizon: int,
        hours_back: int,
        datum: str = "MSL",
        city: str | None = None,
        dem_path: str | None = None,
        auto_dem: bool = True,
        provider: str | None = None,
    ) -> dict:
        city_key, cfg = self._resolve_city_config(city=city, provider=provider, station=station)
        df, fetch_meta = self.fetch_latest_series(
            cfg=cfg,
            station_override=station,
            hours_back=hours_back,
            datum=datum,
        )

        preds, forecast_mode_used, forecast_note = self._forecast(cfg=cfg, df=df, horizon=horizon)
        dem_path_resolved = self._resolve_dem_path(cfg=cfg, dem_path=dem_path, auto_dem=auto_dem)

        last_obs_ts = pd.Timestamp(df["timestamp"].iloc[-1])
        forecast_timestamps = [last_obs_ts + pd.Timedelta(hours=i) for i in range(1, horizon + 1)]
        forecast_points = [
            {"timestamp_utc": ts.isoformat(), "sea_level_m": float(val), "hour_ahead": i}
            for i, (ts, val) in enumerate(zip(forecast_timestamps, preds.tolist()), start=1)
        ]

        peak = float(np.max(preds))
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "city": city_key,
            "display_name": cfg.get("display_name", city_key or fetch_meta["station_ref"]),
            "provider": cfg.get("provider"),
            "provider_label": cfg.get("provider_label"),
            "support_tier": cfg.get("support_tier"),
            "proxy_mode": cfg.get("proxy_mode"),
            "station": fetch_meta["station_ref"],
            "datum": datum,
            "units": "meters",
            "city_notes": cfg.get("notes"),
            "model": {
                "type": self.metadata.get("model_type", "baseline")
                if forecast_mode_used == "model_recursive"
                else "baseline_tide_persistence",
                "lookback_hours": int(self.metadata.get("lookback_hours", 24)) if self.metadata else 24,
                "requested_forecast_mode": cfg.get("forecast_mode", "tide_persistence"),
                "forecast_mode_used": forecast_mode_used,
                "forecast_note": forecast_note,
            },
            "source": {
                "status": fetch_meta.get("status", "ok"),
                "note": fetch_meta.get("note"),
                "source_value_column": fetch_meta.get("source_value_column"),
                "observation_delay_hours": fetch_meta.get("observation_delay_hours"),
            },
            "history": {
                "hours_back": hours_back,
                "observations_used": int(len(df)),
                "last_observation_utc": last_obs_ts.isoformat(),
            },
            "horizon_hours": horizon,
            "forecast_values_m": [float(v) for v in preds.tolist()],
            "forecast": forecast_points,
            "peak_prediction_m": peak,
            "dem_path": dem_path_resolved,
        }

        if dem_path_resolved and Path(dem_path_resolved).exists():
            scenarios = []
            city_part = city_key if city_key else _slugify_station_ref(cfg.get("provider", "unknown"), fetch_meta["station_ref"])
            out_dir = Path(f"Backend/sea_level_risk/outputs/realtime/{city_part}")
            out_dir.mkdir(parents=True, exist_ok=True)
            for delta in DEFAULT_SCENARIOS_M:
                level = peak + delta
                name = f"plus_{int(delta * 100)}cm"
                geojson = out_dir / f"flood_{name}.geojson"
                flood = dem_to_flood_polygon(dem_path_resolved, level, str(geojson))
                scenarios.append(
                    {
                        "scenario": name,
                        "scenario_water_level_m": level,
                        "flood_area_m2": flood["flood_area_m2"],
                        "flood_ratio": flood["flood_ratio"],
                        "component_count": flood.get("component_count", 0),
                        "candidate_flood_pixels": flood.get("candidate_flood_pixels", 0),
                        "land_pixels": flood.get("land_pixels", 0),
                        "processing_mode": flood.get("processing_mode", "unknown"),
                        "risk_level": _risk_label(flood["flood_ratio"]),
                        "geojson": str(geojson),
                    }
                )
            payload["scenarios"] = scenarios

        return payload

    def get_hotspots(
        self,
        city: str | None,
        station: str | None,
        limit: int = 10,
        horizon: int = 6,
        hours_back: int = 96,
        datum: str = "MSL",
        provider: str | None = None,
    ) -> dict:
        city_key, cfg = self._resolve_city_config(city=city, provider=provider, station=station)
        station_ref = station or cfg.get("station_id") or cfg.get("station_code") or "unknown"
        city_part = city_key if city_key else _slugify_station_ref(cfg.get("provider", "unknown"), station_ref)

        out_dir = Path(f"Backend/sea_level_risk/outputs/realtime/{city_part}")
        hotspot_geojson = out_dir / "hotspots.geojson"
        hotspot_csv = out_dir / "hotspots.csv"

        if not hotspot_geojson.exists():
            pred = self.predict(
                city=city_key,
                station=station_ref,
                horizon=horizon,
                hours_back=hours_back,
                datum=datum,
                dem_path=None,
                auto_dem=True,
                provider=cfg.get("provider"),
            )
            scenarios = pred.get("scenarios", [])
            if scenarios:
                build_hotspots_from_scenarios(
                    scenario_records=scenarios,
                    out_geojson=str(hotspot_geojson),
                    out_csv=str(hotspot_csv),
                    top_n=max(limit, 20),
                )

        if hotspot_geojson.exists():
            data = gpd.read_file(hotspot_geojson)
            if data.empty:
                return {"city": city_key, "station": station_ref, "source": "hotspots.geojson", "count": 0, "hotspots": []}

            data = data.sort_values("priority_score", ascending=False).head(limit)
            records = data.drop(columns=["geometry"]).to_dict(orient="records")
            return {
                "city": city_key,
                "station": station_ref,
                "provider": cfg.get("provider"),
                "support_tier": cfg.get("support_tier"),
                "source": "priority_engine",
                "count": int(len(records)),
                "hotspots": records,
                "hotspots_geojson": str(hotspot_geojson),
                "hotspots_csv": str(hotspot_csv),
            }

        return {
            "city": city_key,
            "station": station_ref,
            "provider": cfg.get("provider"),
            "support_tier": cfg.get("support_tier"),
            "source": "none",
            "count": 0,
            "hotspots": [],
        }


def create_app(model_path: str, metadata_path: str, dem_path: str | None = None) -> Flask:
    app = Flask(__name__)
    CORS(app)

    service = RealtimeService(model_path=model_path, metadata_path=metadata_path, default_dem_path=dem_path)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/realtime/cities")
    def cities():
        return jsonify(service.city_registry)

    @app.get("/realtime/forecast")
    def realtime_forecast():
        city = request.args.get("city")
        station = request.args.get("station")
        provider = request.args.get("provider")
        horizon = int(request.args.get("horizon", 6))
        hours_back = int(request.args.get("hours_back", 96))
        datum = request.args.get("datum", "MSL")
        dem = request.args.get("dem")
        auto_dem = request.args.get("auto_dem", "1") not in {"0", "false", "False"}

        try:
            result = service.predict(
                city=city,
                station=station,
                provider=provider,
                horizon=horizon,
                hours_back=hours_back,
                datum=datum,
                dem_path=dem,
                auto_dem=auto_dem,
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.get("/realtime/hotspots")
    def realtime_hotspots():
        city = request.args.get("city")
        station = request.args.get("station")
        provider = request.args.get("provider")
        limit = int(request.args.get("limit", 10))
        horizon = int(request.args.get("horizon", 6))
        hours_back = int(request.args.get("hours_back", 96))
        datum = request.args.get("datum", "MSL")

        try:
            result = service.get_hotspots(
                city=city,
                station=station,
                provider=provider,
                limit=limit,
                horizon=horizon,
                hours_back=hours_back,
                datum=datum,
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def main():
    parser = argparse.ArgumentParser(description="Realtime sea-level forecast API")
    parser.add_argument("--model", default="Backend/sea_level_risk/outputs/sea_level_axial_lstm.keras")
    parser.add_argument("--metadata", default="Backend/sea_level_risk/outputs/metadata.json")
    parser.add_argument("--dem", default="data/honolulu_dem.tif")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    dem_path = args.dem if Path(args.dem).exists() else None
    app = create_app(model_path=args.model, metadata_path=args.metadata, dem_path=dem_path)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

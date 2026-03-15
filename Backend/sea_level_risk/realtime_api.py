import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model

from .city_registry import load_city_registry
from .dem_provider import ensure_dem_for_station
from .download_data import _fetch_noaa_chunk
from .forecast import recursive_forecast_with_loaded_model
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


class RealtimeService:
    def __init__(self, model_path: str, metadata_path: str, default_dem_path: str | None = None):
        import json

        self.model = load_model(model_path, compile=False)
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8-sig"))
        self.default_dem_path = default_dem_path
        self.city_registry = load_city_registry()

    def _fetch_window(self, station: str, end_ts: datetime, hours_back: int, datum: str) -> pd.DataFrame:
        begin = (end_ts - timedelta(hours=hours_back)).strftime("%Y%m%d")
        end = end_ts.strftime("%Y%m%d")
        return _fetch_noaa_chunk(
            station=station,
            begin_date=begin,
            end_date=end,
            product="hourly_height",
            datum=datum,
            units="metric",
            time_zone="gmt",
        ).sort_values("timestamp")

    def fetch_latest_series(self, station: str, hours_back: int = 72, datum: str = "MSL") -> pd.DataFrame:
        now_utc = datetime.now(timezone.utc)

        try:
            df = self._fetch_window(station=station, end_ts=now_utc, hours_back=hours_back, datum=datum)
            if not df.empty:
                return df
        except Exception:
            pass

        fallback_end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        df = self._fetch_window(station=station, end_ts=fallback_end, hours_back=hours_back, datum=datum)
        if df.empty:
            raise RuntimeError("No NOAA data returned for realtime and fallback windows.")
        return df

    def _resolve_city_station_dem(self, city: str | None, station: str | None, dem_path: str | None, auto_dem: bool) -> tuple[str, str | None, str | None]:
        resolved_city = city
        resolved_station = station
        resolved_dem = dem_path

        if city:
            city_key = city.strip().lower()
            if city_key not in self.city_registry:
                raise ValueError(f"Unknown city '{city}'. Available: {list(self.city_registry.keys())}")
            city_cfg = self.city_registry[city_key]
            resolved_city = city_key
            resolved_station = resolved_station or city_cfg.get("station")
            resolved_dem = resolved_dem or city_cfg.get("dem_path")

        if not resolved_station:
            resolved_station = "1612340"

        if not resolved_dem:
            resolved_dem = self.default_dem_path

        if (not resolved_dem) and auto_dem:
            resolved_dem = ensure_dem_for_station(resolved_station)

        return resolved_city, resolved_station, resolved_dem

    def predict(
        self,
        station: str | None,
        horizon: int,
        hours_back: int,
        datum: str = "MSL",
        city: str | None = None,
        dem_path: str | None = None,
        auto_dem: bool = True,
    ) -> dict:
        city_key, station_id, dem_path_resolved = self._resolve_city_station_dem(city, station, dem_path, auto_dem)

        df = self.fetch_latest_series(station=station_id, hours_back=hours_back, datum=datum)
        recent = df["sea_level"].to_numpy(dtype=np.float32)

        preds = recursive_forecast_with_loaded_model(
            model=self.model,
            metadata=self.metadata,
            recent_values=recent,
            horizon_hours=horizon,
        )

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
            "station": station_id,
            "datum": datum,
            "units": "meters",
            "model": {
                "type": self.metadata.get("model_type", "unknown"),
                "lookback_hours": int(self.metadata.get("lookback_hours", 24)),
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
            city_part = city_key if city_key else station_id
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
    ) -> dict:
        city_key, station_id, _ = self._resolve_city_station_dem(city, station, dem_path=None, auto_dem=False)
        city_part = city_key if city_key else station_id

        out_dir = Path(f"Backend/sea_level_risk/outputs/realtime/{city_part}")
        hotspot_geojson = out_dir / "hotspots.geojson"
        hotspot_csv = out_dir / "hotspots.csv"

        if not hotspot_geojson.exists():
            pred = self.predict(
                city=city_key,
                station=station_id,
                horizon=horizon,
                hours_back=hours_back,
                datum=datum,
                dem_path=None,
                auto_dem=True,
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
                return {"city": city_key, "station": station_id, "source": "hotspots.geojson", "count": 0, "hotspots": []}

            data = data.sort_values("priority_score", ascending=False).head(limit)
            records = data.drop(columns=["geometry"]).to_dict(orient="records")
            return {
                "city": city_key,
                "station": station_id,
                "source": "priority_engine",
                "count": int(len(records)),
                "hotspots": records,
                "hotspots_geojson": str(hotspot_geojson),
                "hotspots_csv": str(hotspot_csv),
            }

        return {
            "city": city_key,
            "station": station_id,
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
        horizon = int(request.args.get("horizon", 6))
        hours_back = int(request.args.get("hours_back", 96))
        datum = request.args.get("datum", "MSL")
        dem = request.args.get("dem")
        auto_dem = request.args.get("auto_dem", "1") not in {"0", "false", "False"}

        try:
            result = service.predict(
                city=city,
                station=station,
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
        limit = int(request.args.get("limit", 10))
        horizon = int(request.args.get("horizon", 6))
        hours_back = int(request.args.get("hours_back", 96))
        datum = request.args.get("datum", "MSL")

        try:
            result = service.get_hotspots(
                city=city,
                station=station,
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

"""Read existing GIS outputs and inspect them without running a forecast on page load."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, geometry_window
from shapely.geometry import mapping

from .gis import EQUAL_AREA_EPSG, compute_exposure_frames
from .impact import _enrich_exposure_row, _rollup_exposure_rows

SCENARIOS = {"plus_20cm": 0.2, "plus_50cm": 0.5, "plus_100cm": 1.0}
REPO_ROOT = Path(__file__).resolve().parents[2]


class AnalysisNotFound(ValueError):
    pass


class AnalysisChanged(ValueError):
    pass


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def _iso_time(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


class ExplorerService:
    def __init__(self, root: Path = REPO_ROOT, live_provider=None):
        self.root = Path(root).resolve()
        self.outputs = self.root / "Backend/sea_level_risk/outputs"
        self.registry = _json(self.root / "Backend/sea_level_risk/city_registry.json", {})
        self.exposure_registry = _json(self.root / "Backend/sea_level_risk/exposure_registry.json", {})
        self.live_provider = live_provider
        self._lock = RLock()
        self._refresh_lock = Lock()
        self._cache = OrderedDict()

    def _path(self, value: str | Path) -> Path:
        path = Path(str(value).replace("\\", "/"))
        resolved = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Asset path must be inside the repository.")
        return resolved

    def _city_dir(self, city: str) -> Path:
        if city not in self.registry:
            raise AnalysisNotFound("Unknown city.")
        return self.outputs / "realtime" / city

    def _flood_path(self, city: str, scenario: str) -> Path:
        if scenario not in SCENARIOS:
            raise ValueError("Unsupported scenario.")
        return self._city_dir(city) / f"flood_{scenario}.geojson"

    def _version(self, city: str, scenario: str) -> str:
        paths = [self._flood_path(city, scenario),
                 self._flood_path(city, scenario).with_suffix(".meta.json"),
                 self._city_dir(city) / f"forecast_{scenario}.json"]
        paths += [self._path(layer["path"]) for layer in self.exposure_registry.get(city, [])]
        stamp = [(str(p.relative_to(self.root)), p.stat().st_mtime_ns, p.stat().st_size)
                 for p in paths if p.exists()]
        return hashlib.sha256(json.dumps(stamp).encode()).hexdigest()[:16]

    def _cached(self, key, build):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        value = build()
        with self._lock:
            self._cache[key] = value
            if len(self._cache) > 48:
                self._cache.popitem(last=False)
        return value

    def _read_frame(self, path: Path) -> gpd.GeoDataFrame:
        return self._cached(("file", str(path), path.stat().st_mtime_ns, path.stat().st_size),
                            lambda: gpd.read_file(path))

    def catalog(self) -> dict:
        cities = []
        for key, cfg in self.registry.items():
            available = [name for name in SCENARIOS if self._flood_path(key, name).exists()]
            cities.append({
                "id": key, "name": cfg.get("display_name", key),
                "provider": cfg.get("provider_label"), "station": cfg.get("station_id") or cfg.get("station_code"),
                "position": [cfg.get("lat"), cfg.get("lon")],
                "support_tier": cfg.get("support_tier"), "proxy_mode": cfg.get("proxy_mode"),
                "scenarios": available,
            })
        return {"cities": cities, "scenarios": [
            {"id": name, "label": f"+{int(delta * 100)} cm", "delta_m": delta}
            for name, delta in SCENARIOS.items()
        ]}

    def _frame(self, city: str, scenario: str) -> gpd.GeoDataFrame:
        path = self._flood_path(city, scenario)
        if not path.exists():
            raise AnalysisNotFound("This city has no saved map for the selected scenario.")
        frame = self._read_frame(path)
        if frame.crs is None:
            raise ValueError("Flood polygons have no declared CRS.")
        frame = frame.to_crs(4326).explode(index_parts=False).reset_index(drop=True)
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
        if not frame.empty:
            frame.geometry = frame.geometry.make_valid()
        frame["component_id"] = range(1, len(frame) + 1)
        frame["area_m2"] = frame.to_crs(EQUAL_AREA_EPSG).area
        return frame

    def _exposure(self, city: str, flood: gpd.GeoDataFrame, include_features=False):
        rows, statuses, features = [], [], []
        if not flood.empty:
            flood = gpd.GeoDataFrame(geometry=[flood.geometry.union_all()], crs=flood.crs)
        for cfg in self.exposure_registry.get(city, []):
            path = self._path(cfg["path"])
            status = {"id": cfg["name"], "name": cfg.get("display_name", cfg["name"]),
                      "category": cfg.get("category"), "status": "missing"}
            if path.exists():
                try:
                    layer = self._read_frame(path)
                    result = compute_exposure_frames(
                        flood, layer, cfg["name"], value_field=cfg.get("value_field"),
                        weight_by_area=bool(cfg.get("weight_by_area")),
                    )
                    rows.append(_enrich_exposure_row(result, cfg))
                    status["status"] = "ready"
                    if include_features and not flood.empty and not layer.empty:
                        if layer.crs != flood.crs:
                            layer = layer.to_crs(flood.crs)
                        footprint = flood.geometry.union_all()
                        candidates = layer.iloc[layer.sindex.query(footprint, predicate="intersects")].copy()
                        candidates.geometry = candidates.geometry.intersection(footprint)
                        candidates = candidates[~candidates.geometry.is_empty].to_crs(4326)
                        for _, row in candidates.head(max(0, 2500 - len(features))).iterrows():
                            features.append({"type": "Feature", "geometry": mapping(row.geometry), "properties": {
                                "layer": cfg["name"], "category": cfg.get("category"),
                                "name": str(row.get("name") or cfg.get("display_name", cfg["name"])),
                            }})
                except (ValueError, RuntimeError, OSError, KeyError):
                    status["status"] = "unavailable"
            statuses.append(status)
        rollup = _rollup_exposure_rows(rows)
        metrics = {
            "roads_m": rollup["affected_road_length_m"] if any(r["category"] == "transport" for r in rows) else None,
            "sites": rollup["affected_site_count_total"] if any(r["metric"] == "points" for r in rows) else None,
            "population": rollup["population_affected_estimate"] if any(r["category"] == "population" for r in rows) else None,
            "vulnerable_population": rollup["high_vulnerability_population_affected_estimate"] if any(r["category"] == "vulnerability" for r in rows) else None,
        }
        return {"metrics": metrics, "rows": rows, "layers": statuses,
                "features": {"type": "FeatureCollection", "features": features},
                "features_truncated": len(features) >= 2500}

    def analysis(self, city: str, scenario: str) -> dict:
        version = self._version(city, scenario)
        return self._cached(("analysis", city, scenario, version), lambda: self._analysis(city, scenario, version))

    def _analysis(self, city: str, scenario: str, version: str) -> dict:
        frame = self._frame(city, scenario)
        path = self._flood_path(city, scenario)
        meta = _json(path.with_suffix(".meta.json"), {})
        exposure = self._exposure(city, frame, include_features=True)
        # Area ranking is deterministic for one scenario and does not imply a flood probability.
        ranks = {int(row.component_id): index + 1 for index, row in
                 enumerate(frame.sort_values("area_m2", ascending=False).itertuples())}
        features = []
        for row in frame.itertuples():
            point = row.geometry.representative_point()
            properties = {"component_id": int(row.component_id), "area_m2": float(row.area_m2),
                          "rank": ranks[int(row.component_id)], "scenario": scenario,
                          "position": [float(point.y), float(point.x)]}
            features.append({"type": "Feature", "id": int(row.component_id),
                             "geometry": mapping(row.geometry), "properties": properties})
        snapshot = _json(self._city_dir(city) / f"forecast_{scenario}.json", {})
        # Old polygon metadata does not establish the date of a NOAA observation.
        forecast = snapshot.get("forecast", [])
        return {
            "city": city, "name": self.registry[city].get("display_name", city), "scenario": scenario,
            "version": version, "data_mode": "saved_analysis", "saved_at_utc": _iso_time(path),
            "observation_utc": snapshot.get("history", {}).get("last_observation_utc"),
            "provider": self.registry[city].get("provider_label"),
            "source_status": snapshot.get("source", {}).get("status", "not_recorded"),
            "source_note": snapshot.get("source", {}).get("note"),
            "datum": snapshot.get("datum"), "vertical_alignment": "unverified",
            "scenario_water_level_m": meta.get("scenario_water_level_m"),
            "processing_mode": meta.get("processing_mode", "saved_polygon"),
            "metrics": {"area_m2": float(frame["area_m2"].sum()), "components": len(frame),
                        "flood_ratio": meta.get("flood_ratio"), **exposure["metrics"]},
            "flood": {"type": "FeatureCollection", "features": features},
            "exposure": exposure, "forecast": forecast, "model": snapshot.get("model"),
            "bounds": frame.total_bounds.tolist() if not frame.empty else None,
            "hotspots": [feature for feature in sorted(features, key=lambda f: f["properties"]["rank"])[:10]],
        }

    def inspect(self, city: str, scenario: str, component_id: int, version: str | None = None) -> dict:
        current = self._version(city, scenario)
        if version and version != current:
            raise AnalysisChanged("Map changed. Reload the scenario before selecting a region.")
        return self._cached(("inspect", city, scenario, component_id, current),
                            lambda: self._inspect(city, scenario, component_id))

    def _inspect(self, city: str, scenario: str, component_id: int) -> dict:
        frame = self._frame(city, scenario)
        selected = frame[frame["component_id"] == component_id]
        if selected.empty:
            raise AnalysisNotFound("Flood region was not found.")
        geometry = selected.geometry.iloc[0]
        exposure = self._exposure(city, selected, include_features=True)
        comparison = []
        for name in SCENARIOS:
            if not self._flood_path(city, name).exists():
                continue
            other = self._frame(city, name)
            intersection = geometry.intersection(other.geometry.union_all())
            area = gpd.GeoSeries([intersection], crs=4326).to_crs(EQUAL_AREA_EPSG).area.iloc[0]
            comparison.append({"scenario": name, "overlap_area_m2": float(area)})
        return {"component_id": component_id, "scenario": scenario, "area_m2": float(selected["area_m2"].iloc[0]),
                "exposure": exposure, "elevation": self._elevation(city, scenario, geometry),
                "comparison": comparison, "geometry": mapping(geometry), "bounds": list(geometry.bounds)}

    def _elevation(self, city, scenario, geometry):
        meta = _json(self._flood_path(city, scenario).with_suffix(".meta.json"), {})
        if not meta.get("dem_path"):
            return None
        path = self._path(meta["dem_path"])
        if not path.exists():
            return None
        with rasterio.open(path) as src:
            geom = gpd.GeoSeries([geometry], crs=4326).to_crs(src.crs).iloc[0]
            try:
                window = geometry_window(src, [mapping(geom)])
            except rasterio.errors.WindowError:
                return None
            values = src.read(1, window=window, masked=True)
            inside = geometry_mask([mapping(geom)], values.shape, src.window_transform(window), invert=True)
            valid = inside & ~np.ma.getmaskarray(values) & np.isfinite(values.data) & (values.data > 0)
            sample = values.data[valid]
            if not sample.size:
                return None
            return {"min_m": float(sample.min()), "mean_m": float(sample.mean()),
                    "max_m": float(sample.max()), "samples": int(sample.size), "reference": "DEM vertical reference"}

    def evidence(self, city: str) -> dict:
        self._city_dir(city)
        reports = []
        for path in self.outputs.glob(f"backtests*/{city}/summary.json"):
            data = _json(path, {})
            reports.append({"id": path.parent.parent.name, "source_mode": data.get("source_mode"),
                            "horizon_hours": data.get("horizon_hours"), "windows": data.get("n_windows"),
                            "results": [{k: row.get(k) for k in [
                                "forecaster", "mae_m", "rmse_m", "bias_m", "p10_p90_coverage",
                                "n_forecast_points", "start_target_utc", "end_target_utc",
                            ]} for row in data.get("summaries", [])]})
        return {"city": city, "reports": sorted(reports, key=lambda r: r["id"])}

    def refresh(self, city: str, scenario: str, horizon: int = 6) -> dict:
        self._flood_path(city, scenario)
        if horizon not in range(1, 25):
            raise ValueError("Forecast horizon must be between 1 and 24 hours.")
        if self.live_provider is None:
            raise RuntimeError("Live forecasts are not configured on this server.")
        if not self._refresh_lock.acquire(blocking=False):
            raise AnalysisChanged("An observation refresh is already running. Try again shortly.")
        try:
            payload = self.live_provider(city=city, station=None, horizon=horizon, hours_back=96,
                                         datum="MSL", auto_dem=False, scenario_names=[scenario])
            out = self._city_dir(city) / f"forecast_{scenario}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            temporary = out.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(out)
            with self._lock:
                self._cache.clear()
        finally:
            self._refresh_lock.release()
        return {"city": city, "source_status": payload.get("source", {}).get("status"),
                "observation_utc": payload.get("history", {}).get("last_observation_utc")}

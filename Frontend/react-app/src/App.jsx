import React, { useEffect, useRef, useState } from "react";
import {
  NavLink,
  Navigate,
  Route,
  Routes,
  useLocation,
  useSearchParams,
} from "react-router-dom";
import {
  Activity,
  ArrowDownToLine,
  ArrowUpRight,
  Check,
  ChevronDown,
  Compass,
  Crosshair,
  Layers3,
  MapPin,
  RefreshCw,
  Share2,
  Waves,
  X,
} from "lucide-react";
import { request, downloadJson, number, dateLabel, scenarioLabel } from "./api";
import FloodMap from "./components/FloodMap";
import {
  Spotlight,
  Operations,
  Exposure,
  Hotspots,
  Forecast,
  Compare,
} from "./pages/AnalysisPages";

const PAGES = [
  ["/", "City spotlight"],
  ["/operations", "Operations"],
  ["/exposure", "Exposure"],
  ["/hotspots", "Hotspots"],
  ["/forecast", "Forecast"],
  ["/compare", "Compare"],
];

function Inspector({ selected, inspecting, error, onClose, onExport }) {
  return (
    <aside className="inspector" aria-label="Selected flood region">
      <div className="inspector-heading">
        <div>
          <span className="overline">REGION INSPECTOR</span>
          <h3>
            {selected
              ? `Region ${String(selected.component_id).padStart(2, "0")}`
              : "Reading region"}
          </h3>
        </div>
        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Close region inspector"
        >
          <X size={17} />
        </button>
      </div>
      {inspecting ? (
        <p role="status" className="subtle">
          Calculating spatial intersections...
        </p>
      ) : error ? (
        <p role="alert">{error}</p>
      ) : (
        selected && (
          <>
            <div className="inspector-metrics">
              <div>
                <span>Footprint</span>
                <strong>
                  {number(selected.area_m2 / 10000, 2)} <small>ha</small>
                </strong>
              </div>
              <div>
                <span>Roads exposed</span>
                <strong>
                  {number(selected.exposure.metrics.roads_m)} <small>m</small>
                </strong>
              </div>
              <div>
                <span>Priority sites</span>
                <strong>{number(selected.exposure.metrics.sites)}</strong>
              </div>
            </div>
            {selected.elevation && (
              <p className="inspector-detail">
                Ground elevation{" "}
                <strong>{number(selected.elevation.mean_m, 2)} m</strong>
                <span>DEM mean; vertical alignment unverified</span>
              </p>
            )}
            <div className="region-overlap">
              <span className="overline">THIS REGION ACROSS SCENARIOS</span>
              {selected.comparison.map((item) => (
                <div key={item.scenario}>
                  <span>{scenarioLabel(item.scenario)}</span>
                  <meter
                    min="0"
                    max={Math.max(
                      selected.area_m2,
                      ...selected.comparison.map((v) => v.overlap_area_m2),
                    )}
                    value={item.overlap_area_m2}
                  />
                  <strong>{number(item.overlap_area_m2 / 10000, 2)} ha</strong>
                </div>
              ))}
            </div>
            <button className="text-button" onClick={onExport}>
              <ArrowDownToLine size={14} /> Export selected region
            </button>
          </>
        )
      )}
    </aside>
  );
}

export default function App() {
  const [search, setSearch] = useSearchParams();
  const { pathname } = useLocation();
  const [catalog, setCatalog] = useState(null);
  const [catalogError, setCatalogError] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [selected, setSelected] = useState(null);
  const [inspecting, setInspecting] = useState(false);
  const [inspectionError, setInspectionError] = useState("");
  const [layersOpen, setLayersOpen] = useState(false);
  const [layers, setLayers] = useState({
    flood: true,
    infrastructure: false,
    hotspots: true,
  });
  const [baseMap, setBaseMap] = useState("streets");
  const [refreshing, setRefreshing] = useState(false);
  const [toast, setToast] = useState("");
  const [fitRequest, setFitRequest] = useState(0);
  const cache = useRef(new Map());
  const cityId = search.get("city") || "boston";
  const scenario = search.get("scenario") || "plus_50cm";
  const regionId = Number(search.get("region")) || null;
  const city = catalog?.cities.find((item) => item.id === cityId);
  const analysis =
    data?.city === cityId && data?.scenario === scenario ? data : null;
  const pageName =
    PAGES.find((item) => item[0] === pathname)?.[1] || "City spotlight";

  useEffect(() => {
    const controller = new AbortController();
    request("/catalog", { signal: controller.signal })
      .then(setCatalog)
      .catch((e) => {
        if (e.name !== "AbortError")
          setCatalogError(
            "Cannot reach the analysis server. Start the Explorer API on port 8100.",
          );
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!catalog) return;
    const controller = new AbortController();
    const key = `${cityId}:${scenario}:${revision}`;
    setError("");
    setSelected(null);
    if (cache.current.has(key)) {
      setData(cache.current.get(key));
      setLoading(false);
      return () => controller.abort();
    }
    setLoading(true);
    request(`/cities/${cityId}/analysis?scenario=${scenario}`, {
      signal: controller.signal,
    })
      .then((value) => {
        cache.current.set(key, value);
        setData(value);
      })
      .catch((e) => {
        if (e.name !== "AbortError") {
          setError(e.message);
          setData(null);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [catalog, cityId, scenario, revision]);

  useEffect(() => {
    setSelected(null);
    setInspectionError("");
    if (!regionId || !analysis) {
      setInspecting(false);
      return;
    }
    const controller = new AbortController();
    setInspecting(true);
    request(
      `/cities/${cityId}/components/${regionId}?scenario=${scenario}&version=${analysis.version}`,
      { signal: controller.signal },
    )
      .then(setSelected)
      .catch((e) => {
        if (e.name !== "AbortError") setInspectionError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setInspecting(false);
      });
    return () => controller.abort();
  }, [cityId, scenario, regionId, analysis]);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 5000);
    return () => clearTimeout(timer);
  }, [toast]);

  function updateQuery(values, clearRegion = false) {
    setSearch((current) => {
      const next = new URLSearchParams(current);
      if (clearRegion) next.delete("region");
      Object.entries(values).forEach(([key, value]) =>
        value == null ? next.delete(key) : next.set(key, value),
      );
      return next;
    });
  }

  async function refresh() {
    if (refreshing) return;
    setRefreshing(true);
    try {
      await request(`/cities/${cityId}/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario, horizon: 6 }),
      });
      cache.current.clear();
      updateQuery({ region: null });
      setRevision((value) => value + 1);
      setToast(
        "Forecast refreshed. See observation time on the Forecast page.",
      );
    } catch (e) {
      setToast(e.message);
    } finally {
      setRefreshing(false);
    }
  }

  function exportMap() {
    if (analysis) downloadJson(analysis.flood, `${cityId}_${scenario}.geojson`);
  }

  const selectRegion = (id) => updateQuery({ region: id });
  const context = {
    analysis,
    city,
    cityId,
    scenario,
    onSelect: selectRegion,
    search: search.toString(),
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <NavLink
          className="brand"
          to={`/?${search}`}
          aria-label="Coastal Flood Risk home"
        >
          <span className="brand-icon">
            <Waves size={24} />
          </span>
          <span>
            COASTAL<span className="brand-secondary">FLOOD RISK</span>
          </span>
        </NavLink>
        <div className="header-context">
          <span className="status-dot" /> GEOSPATIAL EXPLORER{" "}
          <span className="version">01</span>
        </div>
        <button
          className="share-button"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(window.location.href);
              setToast("Link copied with city, scenario and selected region.");
            } catch {
              setToast("Use the browser address to share this view.");
            }
          }}
        >
          <Share2 size={15} />
          <span>Share view</span>
        </button>
      </header>
      <nav className="page-nav" aria-label="Analysis pages">
        {PAGES.map(([path, label]) => (
          <NavLink
            key={path}
            end={path === "/"}
            to={{ pathname: path, search: `?${search}` }}
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="control-bar">
        <div className="city-control">
          <MapPin size={17} />
          <label>
            <span>MONITORING LOCATION</span>
            <select
              aria-label="City"
              disabled={refreshing || !catalog}
              value={cityId}
              onChange={(event) => {
                const chosen = catalog.cities.find(
                  (c) => c.id === event.target.value,
                );
                updateQuery(
                  {
                    city: chosen.id,
                    scenario: chosen.scenarios.includes(scenario)
                      ? scenario
                      : chosen.scenarios[0],
                  },
                  true,
                );
              }}
            >
              {catalog ? (
                catalog.cities
                  .filter((item) => item.scenarios.length)
                  .map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))
              ) : (
                <option>Loading locations...</option>
              )}
            </select>
          </label>
          <ChevronDown size={14} />
        </div>
        <div className="scenario-control">
          <span className="control-label">WATER-LEVEL SCENARIO</span>
          <div role="group" aria-label="Water-level scenario">
            {["plus_20cm", "plus_50cm", "plus_100cm"].map((name) => (
              <button
                key={name}
                disabled={!city?.scenarios.includes(name) || refreshing}
                className={scenario === name ? "active" : ""}
                aria-pressed={scenario === name}
                onClick={() => updateQuery({ scenario: name }, true)}
              >
                {scenarioLabel(name)}
              </button>
            ))}
          </div>
        </div>
        <button
          className="refresh-button"
          onClick={refresh}
          disabled={refreshing || !analysis}
        >
          <RefreshCw size={15} className={refreshing ? "spin" : ""} />
          <span>
            {refreshing ? "Fetching observations..." : "Refresh observations"}
          </span>
        </button>
      </div>
      <main className="workspace">
        <section className="analysis-pane" aria-label={pageName}>
          <div className="section-index">
            <span>
              <span className="status-dot" />{" "}
              {analysis?.forecast.length ? "SAVED FORECAST" : "SAVED ANALYSIS"}
            </span>
            <span>
              {analysis ? dateLabel(analysis.saved_at_utc) : "LOCAL DATA"}
            </span>
          </div>
          {catalogError ? (
            <div className="empty-state" role="alert">
              <Activity size={26} />
              <h2>Analysis server offline</h2>
              <p>{catalogError}</p>
            </div>
          ) : error ? (
            <div className="empty-state" role="alert">
              <h2>Map unavailable</h2>
              <p>{error}</p>
              <button
                className="button"
                onClick={() => setRevision((v) => v + 1)}
              >
                Try again
              </button>
            </div>
          ) : loading || !analysis ? (
            <div className="page-loading" role="status">
              <span className="overline">{pageName}</span>
              <div className="skeleton skeleton-title" />
              <div className="skeleton skeleton-block" />
              <p>Calculating scenario exposure...</p>
            </div>
          ) : (
            <Routes>
              <Route path="/" element={<Spotlight {...context} />} />
              <Route path="/operations" element={<Operations {...context} />} />
              <Route path="/exposure" element={<Exposure {...context} />} />
              <Route path="/hotspots" element={<Hotspots {...context} />} />
              <Route path="/forecast" element={<Forecast {...context} />} />
              <Route path="/compare" element={<Compare {...context} />} />
              <Route
                path="*"
                element={<Navigate to={`/?${search}`} replace />}
              />
            </Routes>
          )}
          <footer className="pane-footer">
            <Compass size={14} />
            <span>Scenario-based exposure assessment</span>
            <span className="footer-coordinate">
              {city?.position?.map((v) => number(v, 2)).join(", ")}
            </span>
          </footer>
        </section>
        <section className="map-pane" aria-label="Interactive flood map">
          <FloodMap
            analysis={analysis}
            city={city}
            scenario={scenario}
            layers={layers}
            baseMap={baseMap}
            selectedId={regionId}
            fitRequest={fitRequest}
            onSelect={selectRegion}
          />
          <div className="map-caption">
            <span className="map-crosshair">
              <Crosshair size={15} />
            </span>
            <div>
              <strong>{city?.name || "Coastal flood extent"}</strong>
              <span>{scenarioLabel(scenario)} scenario</span>
            </div>
          </div>
          <div className="map-tools">
            <button
              className={layersOpen ? "active" : ""}
              aria-expanded={layersOpen}
              aria-label="Map layers"
              onClick={() => setLayersOpen((v) => !v)}
            >
              <Layers3 size={18} />
            </button>
            <button
              aria-label="Reset map extent"
              onClick={() => {
                updateQuery({ region: null });
                setFitRequest((v) => v + 1);
              }}
            >
              <Crosshair size={18} />
            </button>
            <button
              aria-label="Export flood GeoJSON"
              disabled={!analysis}
              onClick={exportMap}
            >
              <ArrowDownToLine size={18} />
            </button>
          </div>
          {layersOpen && (
            <aside className="layer-menu" aria-label="Map layer controls">
              <h3>Map layers</h3>
              {[
                ["flood", "Flood extent"],
                ["hotspots", "Top regions"],
                ["infrastructure", "Affected infrastructure"],
              ].map(([key, label]) => (
                <label key={key}>
                  <input
                    type="checkbox"
                    checked={layers[key]}
                    onChange={(event) =>
                      setLayers((v) => ({ ...v, [key]: event.target.checked }))
                    }
                  />
                  {label}
                </label>
              ))}
              <div className="basemap-control">
                <button
                  className={baseMap === "streets" ? "active" : ""}
                  onClick={() => setBaseMap("streets")}
                >
                  Streets
                </button>
                <button
                  className={baseMap === "satellite" ? "active" : ""}
                  onClick={() => setBaseMap("satellite")}
                >
                  Satellite
                </button>
              </div>
            </aside>
          )}
          {!regionId && analysis && (
            <div className="map-instruction">
              <MapPin size={14} /> Click a flood region to inspect exposure{" "}
              <ArrowUpRight size={14} />
            </div>
          )}
          {regionId && (
            <Inspector
              selected={selected}
              inspecting={inspecting}
              error={inspectionError}
              onClose={() => updateQuery({ region: null })}
              onExport={() =>
                downloadJson(
                  {
                    type: "FeatureCollection",
                    features: [
                      {
                        type: "Feature",
                        geometry: selected.geometry,
                        properties: {
                          city: cityId,
                          scenario,
                          area_m2: selected.area_m2,
                        },
                      },
                    ],
                  },
                  `${cityId}_region_${regionId}.geojson`,
                )
              }
            />
          )}
          <div className="map-legend">
            <span>
              <i className="legend-flood" /> Flood extent
            </span>
            <span>
              <i className="legend-hotspot" /> Top regions
            </span>
            <span className="legend-unit">WGS 84</span>
          </div>
          {loading && (
            <div className="map-loading" role="status">
              <RefreshCw size={16} className="spin" /> Loading scenario
            </div>
          )}
        </section>
      </main>
      {toast && (
        <div className="toast" role="status">
          <Check size={17} />
          <span>{toast}</span>
          <button
            aria-label="Dismiss notification"
            onClick={() => setToast("")}
          >
            <X size={16} />
          </button>
        </div>
      )}
    </div>
  );
}

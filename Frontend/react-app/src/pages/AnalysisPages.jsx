import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  Database,
  Filter,
  MapPin,
  Search,
  ShieldCheck,
  Waves,
} from "lucide-react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import {
  request,
  number,
  dateLabel,
  scenarioLabel,
  downloadJson,
} from "../api";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
);

export function Stat({ label, value, unit, detail }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>
        {value}
        <small>{unit}</small>
      </strong>
      {detail && <span className="stat-detail">{detail}</span>}
    </div>
  );
}

function Title({ eyebrow, title, children }) {
  return (
    <div className="page-title">
      <span className="overline">{eyebrow}</span>
      <h1>{title}</h1>
      {children}
    </div>
  );
}

function LayerCoverage({ layers }) {
  const available = layers.filter((row) => row.status === "ready").length;
  return (
    <div className="coverage-line">
      <Database size={14} />
      <span>
        {available} / {layers.length} exposure layers available
      </span>
      <span className={available === layers.length ? "coverage-ready" : ""}>
        {available === layers.length ? (
          <CheckCircle2 size={14} />
        ) : (
          "Partial coverage"
        )}
      </span>
    </div>
  );
}

export function Spotlight({ analysis, city, search }) {
  return (
    <div className="page-content spotlight-page">
      <Title
        eyebrow={`STATION ${city?.station || "UNRECORDED"}`}
        title={analysis.name}
      >
        <p className="location-provider">
          <span className="status-dot" />
          {analysis.provider}
        </p>
      </Title>
      <div className="scenario-summary">
        <span className="scenario-number">
          {scenarioLabel(analysis.scenario)}
        </span>
        <div>
          <strong>Water-level scenario</strong>
          <span>Added to the saved forecast peak</span>
        </div>
        <Waves size={27} />
      </div>
      <div className="stat-grid">
        <Stat
          label="Flood footprint"
          value={number(analysis.metrics.area_m2 / 10000, 2)}
          unit="ha"
        />
        <Stat
          label="Flood regions"
          value={number(analysis.metrics.components)}
        />
        <Stat
          label="Roads exposed"
          value={number(analysis.metrics.roads_m)}
          unit="m"
        />
        <Stat
          label="Land inundated"
          value={
            analysis.metrics.flood_ratio == null
              ? "\u2014"
              : number(analysis.metrics.flood_ratio * 100, 3)
          }
          unit="%"
        />
      </div>
      <div className="explore-links">
        <span className="overline">EXPLORE THIS SCENARIO</span>
        <Link to={`/operations?${search}`}>
          <span>
            <ShieldCheck size={18} />
            Operational impact
          </span>
          <ArrowUpRight size={18} />
        </Link>
        <Link to={`/hotspots?${search}`}>
          <span>
            <MapPin size={18} />
            Priority regions
          </span>
          <ArrowUpRight size={18} />
        </Link>
      </div>
      <LayerCoverage layers={analysis.exposure.layers} />
      {city?.proxy_mode && city.proxy_mode !== "direct_station" && (
        <p className="data-footnote">Regional proxy gauge · {city.station}</p>
      )}
    </div>
  );
}

export function Operations({ analysis, search }) {
  const metrics = analysis.metrics;
  return (
    <div className="page-content">
      <Title eyebrow="IMPACT ASSESSMENT" title="Operations">
        <p>
          {scenarioLabel(analysis.scenario)} · {analysis.name}
        </p>
      </Title>
      <div className="stat-grid">
        <Stat
          label="Road network"
          value={number(metrics.roads_m)}
          unit="m"
          detail="Intersecting the flood extent"
        />
        <Stat label="Priority sites" value={number(metrics.sites)} />
        <Stat label="Population estimate" value={number(metrics.population)} />
        <Stat
          label="High-vulnerability estimate"
          value={number(metrics.vulnerable_population)}
        />
      </div>
      <p className="data-footnote">
        Population is estimated by flooded share of census-tract area.
      </p>
      <h2 className="section-title">Infrastructure impact</h2>
      <div className="impact-list">
        {analysis.exposure.rows
          .filter((row) => row.metric !== "people")
          .map((row) => (
            <div key={row.layer}>
              <span>{row.display_name}</span>
              <strong className={row.affected_value > 0 ? "value-active" : ""}>
                {number(
                  row.metric === "length_m"
                    ? row.affected_value / 1000
                    : row.affected_value,
                  row.metric === "length_m" ? 2 : 0,
                )}
                <small>
                  {row.metric === "length_m" ? "km" : row.affected_unit}
                </small>
              </strong>
            </div>
          ))}
      </div>
      <Link className="button full-width" to={`/exposure?${search}`}>
        View exposure breakdown <ArrowRight size={16} />
      </Link>
    </div>
  );
}

export function Exposure({ analysis }) {
  const [query, setQuery] = useState("");
  const [onlyAffected, setOnlyAffected] = useState(false);
  const rows = analysis.exposure.rows.filter(
    (row) =>
      row.display_name.toLowerCase().includes(query.toLowerCase()) &&
      (!onlyAffected || row.affected_value > 0),
  );
  return (
    <div className="page-content">
      <Title eyebrow="SPATIAL INTERSECTIONS" title="Exposure">
        <p>
          {analysis.exposure.rows.length} layers ·{" "}
          {scenarioLabel(analysis.scenario)} footprint
        </p>
      </Title>
      <div className="table-controls">
        <label className="search-input">
          <Search size={16} />
          <input
            aria-label="Search exposure layers"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Find a layer"
          />
        </label>
        <button
          className={onlyAffected ? "filter-button active" : "filter-button"}
          aria-pressed={onlyAffected}
          onClick={() => setOnlyAffected((v) => !v)}
        >
          <Filter size={14} />
          Affected
        </button>
      </div>
      <div className="exposure-table">
        <div className="table-head">
          <span>EXPOSURE LAYER</span>
          <span>AFFECTED</span>
        </div>
        {rows.map((row) => (
          <div className="exposure-row" key={row.layer}>
            <span>
              <strong>{row.display_name}</strong>
              <small>{row.category.replaceAll("_", " ")}</small>
            </span>
            <span
              className={row.affected_value > 0 ? "value-active" : "subtle"}
            >
              <strong>
                {number(row.affected_value, row.metric === "length_m" ? 1 : 0)}
              </strong>
              <small>{row.affected_unit}</small>
            </span>
          </div>
        ))}
        {!rows.length && <p className="subtle">No matching layers.</p>}
      </div>
      <LayerCoverage layers={analysis.exposure.layers} />
      <button
        className="text-button"
        onClick={() =>
          downloadJson(
            {
              city: analysis.city,
              scenario: analysis.scenario,
              layers: analysis.exposure.rows,
            },
            `${analysis.city}_exposure.json`,
          )
        }
      >
        <ArrowDownToLine size={15} />
        Export exposure results
      </button>
    </div>
  );
}

export function Hotspots({ analysis, onSelect }) {
  return (
    <div className="page-content">
      <Title eyebrow="AREA RANKING" title="Priority regions">
        <p>Largest connected footprints · {scenarioLabel(analysis.scenario)}</p>
      </Title>
      <div className="rank-heading">
        <span>REGION</span>
        <span>AREA</span>
      </div>
      <div className="hotspot-list">
        {analysis.hotspots.map((feature) => (
          <button
            key={feature.id}
            onClick={() => onSelect(feature.properties.component_id)}
            aria-label={`Inspect region ${feature.properties.component_id}`}
          >
            <span className="rank-number">
              {String(feature.properties.rank).padStart(2, "0")}
            </span>
            <span className="region-name">
              <strong>
                Region{" "}
                {String(feature.properties.component_id).padStart(2, "0")}
              </strong>
              <span className="rank-bar">
                <i
                  style={{
                    width: `${(feature.properties.area_m2 / analysis.hotspots[0].properties.area_m2) * 100}%`,
                  }}
                />
              </span>
            </span>
            <strong>
              {number(feature.properties.area_m2 / 10000, 2)}
              <small>ha</small>
            </strong>
            <ArrowUpRight size={16} />
          </button>
        ))}
      </div>
      <p className="data-footnote">
        Ranked by footprint area within this scenario.
      </p>
    </div>
  );
}

export function Forecast({ analysis, cityId }) {
  const [evidence, setEvidence] = useState(null);
  const [evidenceError, setEvidenceError] = useState("");
  const [reportId, setReportId] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setEvidence(null);
    setEvidenceError("");
    setReportId("");
    request(`/cities/${cityId}/evidence`, { signal: controller.signal })
      .then((value) => {
        setEvidence(value);
        setReportId(
          value.reports.find((r) => r.id === "backtests_runtime_snapshot_30d")
            ?.id ||
            value.reports[0]?.id ||
            "",
        );
      })
      .catch((e) => {
        if (e.name !== "AbortError") setEvidenceError(e.message);
      });
    return () => controller.abort();
  }, [cityId]);
  const report = evidence?.reports.find((r) => r.id === reportId);
  const points = analysis.forecast;
  const chart = useMemo(
    () => ({
      labels: points.map((p) => `+${p.hour_ahead}h`),
      datasets: [
        {
          label: "P10",
          data: points.map((p) => p.p10_m),
          borderWidth: 0,
          pointRadius: 0,
          backgroundColor: "rgba(88,190,169,0.16)",
          fill: "+1",
        },
        {
          label: "P90",
          data: points.map((p) => p.p90_m),
          borderWidth: 0,
          pointRadius: 0,
        },
        {
          label: "Median",
          data: points.map((p) => p.p50_m ?? p.sea_level_m),
          borderColor: "#6ac6ac",
          backgroundColor: "#6ac6ac",
          pointRadius: 3,
          borderWidth: 2,
          tension: 0.2,
        },
      ],
    }),
    [points],
  );
  const results = report?.results || [];
  return (
    <div className="page-content">
      <Title eyebrow="FORECAST & VALIDATION" title="Water levels">
        <p>Observation: {dateLabel(analysis.observation_utc)}</p>
      </Title>
      {points.length ? (
        <>
          <div className="forecast-chart">
            <Line
              data={chart}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  x: { ticks: { color: "#a5b3b0" }, grid: { display: false } },
                  y: {
                    title: {
                      display: true,
                      text: "Water level (m)",
                      color: "#a5b3b0",
                    },
                    ticks: { color: "#a5b3b0" },
                    grid: { color: "#2c3535" },
                  },
                },
              }}
            />
          </div>
          <span className="data-footnote">
            Median and P10-P90 interval ·{" "}
            {analysis.datum || "Datum not recorded"}
          </span>
        </>
      ) : (
        <div className="forecast-empty">
          <ActivityIcon />
          <strong>No saved forecast</strong>
          <p>Use Refresh observations to fetch a water-level forecast.</p>
        </div>
      )}
      <h2 className="section-title">Backtest evidence</h2>
      {evidenceError ? (
        <p role="alert">{evidenceError}</p>
      ) : evidence?.reports.length ? (
        <>
          <label className="report-select">
            Evaluation run
            <select
              aria-label="Evaluation run"
              value={reportId}
              onChange={(event) => setReportId(event.target.value)}
            >
              {evidence.reports.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id.replaceAll("_", " ")}
                </option>
              ))}
            </select>
          </label>
          <div className="evidence-table">
            <div className="table-head">
              <span>FORECASTER</span>
              <span>MAE / RMSE</span>
            </div>
            {results.map((row) => (
              <div key={row.forecaster}>
                <span>
                  <strong>
                    {row.forecaster === "city_model"
                      ? "City model"
                      : "Tide baseline"}
                  </strong>
                  <small>{number(row.n_forecast_points)} predictions</small>
                </span>
                <strong>
                  {number(row.mae_m == null ? null : row.mae_m * 100, 1)} /{" "}
                  {number(row.rmse_m == null ? null : row.rmse_m * 100, 1)}
                  <small>cm</small>
                </strong>
              </div>
            ))}
          </div>
          <p className="data-footnote">
            {dateLabel(results[0]?.start_target_utc)} to{" "}
            {dateLabel(results[0]?.end_target_utc)} · {report.horizon_hours}h
            horizon · {report.windows} windows
          </p>
        </>
      ) : (
        <p className="subtle">
          {evidence
            ? "No backtest report available."
            : "Loading evaluation runs..."}
        </p>
      )}
      <details className="quality-details">
        <summary>
          Data provenance <Database size={14} />
        </summary>
        <dl>
          <div>
            <dt>Source</dt>
            <dd>{analysis.provider}</dd>
          </div>
          <div>
            <dt>Observation date</dt>
            <dd>{dateLabel(analysis.observation_utc)}</dd>
          </div>
          <div>
            <dt>Source status</dt>
            <dd>{analysis.source_status.replaceAll("_", " ")}</dd>
          </div>
          <div>
            <dt>Vertical datum</dt>
            <dd>{analysis.datum || "Not recorded"}</dd>
          </div>
          <div>
            <dt>DEM alignment</dt>
            <dd>Unverified</dd>
          </div>
          <div>
            <dt>Processing</dt>
            <dd>{analysis.processing_mode.replaceAll("_", " ")}</dd>
          </div>
        </dl>
        {analysis.source_note && <p>{analysis.source_note}</p>}
      </details>
    </div>
  );
}

function ActivityIcon() {
  return <BarChart3 size={25} />;
}

export function Compare({ analysis, cityId, city, scenario }) {
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setResults([]);
    setError("");
    Promise.all(
      city.scenarios.map((name) =>
        name === scenario
          ? Promise.resolve(analysis)
          : request(`/cities/${cityId}/analysis?scenario=${name}`, {
              signal: controller.signal,
            }),
      ),
    )
      .then(setResults)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [cityId, analysis.version]);
  const maxArea = Math.max(1, ...results.map((item) => item.metrics.area_m2));
  return (
    <div className="page-content">
      <Title eyebrow="SCENARIO COMPARISON" title="A changing coastline">
        <p>{analysis.name}</p>
      </Title>
      {error ? (
        <p role="alert">{error}</p>
      ) : !results.length ? (
        <p role="status">Calculating scenarios...</p>
      ) : (
        <>
          <h2 className="section-title">Flood footprint</h2>
          <div className="scenario-comparison">
            {results.map((result) => (
              <div key={result.scenario}>
                <div>
                  <strong>{scenarioLabel(result.scenario)}</strong>
                  <span>{number(result.metrics.area_m2 / 10000, 2)} ha</span>
                </div>
                <div className="comparison-track">
                  <i
                    style={{
                      width: `${(result.metrics.area_m2 / maxArea) * 100}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="compare-table">
            <table>
              <thead>
                <tr>
                  <th>Scenario</th>
                  <th>Roads (m)</th>
                  <th>Sites</th>
                  <th>People*</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result) => (
                  <tr key={result.scenario}>
                    <td>{scenarioLabel(result.scenario)}</td>
                    <td>{number(result.metrics.roads_m)}</td>
                    <td>{number(result.metrics.sites)}</td>
                    <td>{number(result.metrics.population)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="data-footnote">
            *Area-weighted population estimates. Each scenario uses its saved
            forecast peak.
          </p>
        </>
      )}
    </div>
  );
}

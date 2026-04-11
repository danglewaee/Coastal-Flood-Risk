from pathlib import Path
import sys

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .asset_readiness import split_city_keys_by_map_status
    from .city_registry import load_city_registry
    from .operational_summary import build_briefing_markdown
    from .render_2d import SCENARIO_2D_STYLE, build_2d_layers
    from .render_3d import render_3d_flood_map_multi
except ImportError:
    from Backend.sea_level_risk.asset_readiness import split_city_keys_by_map_status
    from Backend.sea_level_risk.city_registry import load_city_registry
    from Backend.sea_level_risk.operational_summary import build_briefing_markdown
    from Backend.sea_level_risk.render_2d import SCENARIO_2D_STYLE, build_2d_layers
    from Backend.sea_level_risk.render_3d import render_3d_flood_map_multi


RISK_COLORS = {
    "low": "#2a9d8f",
    "moderate": "#e9c46a",
    "high": "#f4a261",
    "critical": "#e63946",
}


def risk_badge(level: str) -> str:
    color = RISK_COLORS.get(level, "#6c757d")
    txt = (level or "unknown").upper()
    return f"<span style='background:{color};color:white;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700'>{txt}</span>"


def fetch_payload(api_base: str, city: str, horizon: int, hours_back: int, auto_dem: bool = True) -> dict:
    url = f"{api_base.rstrip('/')}/realtime/forecast"
    params = {
        "city": city,
        "horizon": horizon,
        "hours_back": hours_back,
        "datum": "MSL",
        "auto_dem": 1 if auto_dem else 0,
    }
    resp = requests.get(url, params=params, timeout=180)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def ensure_hotspots(api_base: str, city: str, horizon: int, hours_back: int) -> Path:
    hotspot_geojson = Path("Backend/sea_level_risk/outputs/realtime") / city / "hotspots.geojson"
    if not hotspot_geojson.exists():
        try:
            requests.get(
                f"{api_base.rstrip('/')}/realtime/hotspots",
                params={"city": city, "limit": 20, "horizon": horizon, "hours_back": hours_back, "datum": "MSL"},
                timeout=180,
            )
        except Exception:
            pass
    return hotspot_geojson


def scenario_items_from_payload(payload: dict, selected_scenario: str, show_all: bool) -> list[dict]:
    scenarios = payload.get("scenarios", [])
    if show_all:
        return [
            {
                "scenario": s["scenario"],
                "flood_geojson": s["geojson"],
                "water_level_m": s.get("scenario_water_level_m"),
            }
            for s in scenarios
        ]
    for s in scenarios:
        if s["scenario"] == selected_scenario:
            return [{
                "scenario": s["scenario"],
                "flood_geojson": s["geojson"],
                "water_level_m": s.get("scenario_water_level_m"),
            }]
    return []


def render_header(payload: dict):
    history = payload.get("history", {})
    model_meta = payload.get("model", {})
    source_meta = payload.get("source", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("City", payload.get("display_name") or payload.get("city", "n/a"))
    c2.metric("Station", payload.get("station", "n/a"))
    c3.metric("Peak Prediction (m)", f"{payload.get('peak_prediction_m', float('nan')):.4f}")
    c4.metric("Last Obs (UTC)", history.get("last_observation_utc", "n/a"))
    st.caption(
        f"Provider: {payload.get('provider_label', 'n/a')} | "
        f"Support: {payload.get('support_tier', 'n/a')} | "
        f"Forecast: {model_meta.get('forecast_mode_used', 'n/a')} | "
        f"Features: {model_meta.get('feature_mode', 'n/a')} | "
        f"Obs used: {history.get('observations_used', 'n/a')}"
    )
    if payload.get("city_notes"):
        st.info(payload["city_notes"])
    if source_meta.get("note"):
        st.warning(source_meta["note"])
    if source_meta.get("observation_delay_hours") is not None:
        st.caption(
            f"Source column: {source_meta.get('source_value_column', 'n/a')} | "
            f"Observation delay: {float(source_meta['observation_delay_hours']):.1f} h | "
            f"Source status: {source_meta.get('status', 'n/a')}"
        )


def selected_operational_summary(payload: dict, scenario: str) -> dict | None:
    summaries = payload.get("operational_summaries") or {}
    if scenario in summaries:
        return summaries[scenario]
    return payload.get("operational_summary")


def selected_impact_summary(payload: dict, scenario: str) -> dict | None:
    summaries = payload.get("impact_summaries") or {}
    if scenario in summaries:
        return summaries[scenario]
    return payload.get("impact_summary")


def site_type_rows_from_impact(impact: dict) -> list[dict]:
    rows = []
    for row in impact.get("exposure_summary", []) or []:
        if str(row.get("metric") or "").lower() != "points":
            continue
        value = row.get("affected_value", row.get("affected_point_count", 0))
        try:
            affected_sites = int(round(float(value or 0)))
        except (TypeError, ValueError):
            affected_sites = 0
        rows.append(
            {
                "site_type": row.get("display_name") or row.get("layer") or "Sites",
                "category": row.get("category") or "other",
                "affected_sites": affected_sites,
            }
        )
    rows.sort(key=lambda item: item["affected_sites"], reverse=True)
    return rows


def render_impacted_site_types(impact: dict):
    site_rows = site_type_rows_from_impact(impact)
    if not site_rows:
        return

    st.markdown("**Impacted Site Types**")
    affected_rows = [row for row in site_rows if row["affected_sites"] > 0]
    if affected_rows:
        cols = st.columns(min(4, len(affected_rows)))
        for idx, row in enumerate(affected_rows[:4]):
            cols[idx % len(cols)].metric(row["site_type"], f"{row['affected_sites']:,}")
    else:
        monitored = ", ".join(row["site_type"] for row in site_rows)
        st.caption(f"No monitored priority site types intersect this scenario footprint. Monitored layers: {monitored}.")

    st.dataframe(pd.DataFrame(site_rows), use_container_width=True, hide_index=True)


def render_operational_summary(payload: dict, scenario: str, city_cfg: dict):
    summary = selected_operational_summary(payload, scenario)
    if not summary:
        return

    st.subheader("Operational Summary")
    st.markdown(risk_badge(summary.get("alert_level", "unknown")), unsafe_allow_html=True)
    st.markdown(f"**{summary.get('headline', 'No operational headline available.')}**")
    st.caption(summary.get("summary_text", ""))

    metrics = st.columns(4)
    metrics[0].metric("Alert Level", str(summary.get("alert_level", "n/a")).upper())
    metrics[1].metric("Confidence", str(summary.get("confidence", "n/a")).upper())
    metrics[2].metric(
        "Flood Area",
        "n/a" if summary.get("flood_area_m2") is None else f"{float(summary['flood_area_m2']):,.0f} m2",
    )
    metrics[3].metric(
        "Components",
        "n/a" if summary.get("component_count") is None else f"{int(summary['component_count'])}",
    )

    st.markdown("**Recommended Actions**")
    for item in summary.get("recommended_actions", []):
        st.markdown(f"- {item}")
    st.caption(summary.get("thresholds_note", ""))

    briefing_md = build_briefing_markdown(payload, city_cfg=city_cfg, scenario_name=summary.get("scenario_basis"))
    st.download_button(
        "Download Briefing (Markdown)",
        data=briefing_md,
        file_name=f"{payload.get('city') or 'city'}_{summary.get('scenario_basis') or 'briefing'}_briefing.md",
        mime="text/markdown",
        use_container_width=False,
    )


def render_impact_summary(payload: dict, scenario: str):
    impact = selected_impact_summary(payload, scenario)
    if not impact:
        return

    st.subheader("Priority Hotspots")
    metrics = st.columns(4)
    metrics[0].metric("Hotspots Ranked", f"{int(impact.get('hotspot_count', 0))}")
    metrics[1].metric(
        "Largest Hotspot",
        "n/a" if impact.get("largest_hotspot_area_m2") is None else f"{float(impact['largest_hotspot_area_m2']):,.0f} m2",
    )
    metrics[2].metric(
        "Top-3 Hotspot Area",
        "n/a" if impact.get("top3_hotspot_area_m2") is None else f"{float(impact['top3_hotspot_area_m2']):,.0f} m2",
    )
    metrics[3].metric("Exposure Layers", f"{int(impact.get('exposure_layers_available', 0))}")

    hotspots = impact.get("top_hotspots", [])
    if hotspots:
        hotspot_df = pd.DataFrame(hotspots)[
            ["rank", "scenario", "risk_level", "priority_score", "area_m2", "centroid_lat", "centroid_lon"]
        ].copy()
        hotspot_df["priority_score"] = hotspot_df["priority_score"].map(lambda v: round(float(v), 1))
        hotspot_df["area_m2"] = hotspot_df["area_m2"].map(lambda v: round(float(v), 1))
        st.dataframe(hotspot_df, use_container_width=True, hide_index=True)

    rollup_rows = impact.get("exposure_rollup", [])
    if rollup_rows:
        st.markdown("**Operational Impact Rollup**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            "Road Length Affected",
            "n/a" if impact.get("affected_road_length_m") is None else f"{float(impact['affected_road_length_m'])/1000.0:.2f} km",
        )
        c2.metric("Priority Sites Affected", f"{int(impact.get('affected_site_count_total', 0))}")
        c3.metric(
            "Population Affected",
            "n/a" if impact.get("population_affected_estimate") in {None, 0} else f"{int(impact['population_affected_estimate']):,}",
        )
        c4.metric(
            "High-Vulnerability Population",
            "n/a"
            if impact.get("high_vulnerability_population_affected_estimate") in {None, 0}
            else f"{int(impact['high_vulnerability_population_affected_estimate']):,}",
        )
        c5.metric("Impact Categories", f"{int(impact.get('categories_impacted', 0))}")

        headline_items = impact.get("impact_headline_items", [])
        if headline_items:
            for item in headline_items:
                st.markdown(f"- {item}")

        render_impacted_site_types(impact)

        rollup_df = pd.DataFrame(rollup_rows).copy()
        visible_rollup_df = rollup_df[rollup_df["affected_value"].astype(float) > 0].copy()
        if visible_rollup_df.empty:
            visible_rollup_df = rollup_df
        st.dataframe(
            visible_rollup_df[["display_name", "affected_value", "affected_unit", "layers"]],
            use_container_width=True,
            hide_index=True,
        )

    exposure_rows = impact.get("exposure_summary", [])
    if exposure_rows:
        st.markdown("**Exposure Detail**")
        exposure_df = pd.DataFrame(exposure_rows).copy()
        for col in ["affected_area_m2", "affected_length_m"]:
            if col in exposure_df.columns:
                exposure_df[col] = exposure_df[col].map(lambda v: round(float(v), 1))
        if "affected_value" in exposure_df.columns:
            exposure_df["affected_value"] = exposure_df["affected_value"].map(lambda v: round(float(v), 1) if isinstance(v, float) else v)
        st.dataframe(exposure_df, use_container_width=True, hide_index=True)


def render_single_city(payload: dict, api_base: str, city: str, horizon: int, hours_back: int, scenario: str, show_all: bool, map_mode: str, camera_preset: str, downsample: int, zex: float):
    render_header(payload)
    city_cfg = registry.get(city, {})
    render_operational_summary(payload, scenario=scenario, city_cfg=city_cfg)
    render_impact_summary(payload, scenario=scenario)

    scenarios = payload.get("scenarios", [])
    items = scenario_items_from_payload(payload, selected_scenario=scenario, show_all=show_all)
    hotspot_geojson = ensure_hotspots(api_base, city, horizon, hours_back)

    if scenarios and map_mode in {"2D flood map", "Both"}:
        st.subheader("2D Flood Map")
        layers, view_state = build_2d_layers(items, hotspot_geojson=str(hotspot_geojson) if hotspot_geojson.exists() else None)
        if layers and view_state:
            deck = pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=view_state,
                layers=layers,
                tooltip={
                    "html": "<b>{scenario_label}</b><br/>Risk: {risk_level}<br/>Priority: {priority_score}",
                    "style": {"backgroundColor": "steelblue", "color": "white"},
                },
            )
            st.pydeck_chart(deck, use_container_width=True)
            legend_parts = []
            for key in ["plus_20cm", "plus_50cm", "plus_100cm"]:
                style = SCENARIO_2D_STYLE[key]
                legend_parts.append(
                    f"<span style='display:inline-block;width:12px;height:12px;background:rgba({style['rgba'][0]},{style['rgba'][1]},{style['rgba'][2]},0.8);margin-right:6px;border-radius:2px;'></span>{style['label']}"
                )
            st.markdown(" | ".join(legend_parts), unsafe_allow_html=True)
            if hotspot_geojson.exists():
                st.caption("Red points mark automatically ranked coastal hotspots.")
        else:
            st.warning("2D map could not be built from the current scenario outputs.")

    dem_path = payload.get("dem_path")
    if dem_path and Path(dem_path).exists() and scenarios and map_mode in {"3D terrain map", "Both"} and items:
        st.subheader("3D Terrain Map")
        map_name = "map3d_all_scenarios.html" if show_all else f"map3d_{scenario}.html"
        out_html = Path("Backend/sea_level_risk/outputs/realtime") / city / map_name
        result = render_3d_flood_map_multi(
            dem_path=dem_path,
            scenario_items=items,
            out_html=str(out_html),
            downsample=downsample,
            vertical_exaggeration=zex,
            camera_preset=camera_preset,
        )
        st.success(f"3D map generated: {result['out_html']}")
        st.caption("Legend: +20cm (yellow), +50cm (orange), +100cm (red).")
        html_text = Path(result["out_html"]).read_text(encoding="utf-8")
        components.html(html_text, height=760, scrolling=True)
    elif map_mode in {"3D terrain map", "Both"}:
        st.warning("DEM or scenarios missing for this city.")

    forecast_quantiles = payload.get("forecast_quantiles", [])
    if forecast_quantiles:
        st.subheader("Forecast (P10 / P50 / P90)")
        quantile_df = pd.DataFrame(forecast_quantiles)[["hour_ahead", "p10_m", "p50_m", "p90_m"]]
        st.line_chart(quantile_df, x="hour_ahead", y=["p10_m", "p50_m", "p90_m"])
        if payload.get("model", {}).get("uncertainty", {}).get("method"):
            st.caption(f"Uncertainty method: {payload['model']['uncertainty']['method']}")
    else:
        forecast = payload.get("forecast_values_m", [])
        if forecast:
            st.subheader("Forecast")
            st.line_chart(pd.DataFrame({"hour_ahead": list(range(1, len(forecast) + 1)), "sea_level_m": forecast}), x="hour_ahead", y="sea_level_m")

    forecast_points = payload.get("forecast", [])
    if forecast_points:
        st.subheader("Forecast Timeline")
        st.dataframe(pd.DataFrame(forecast_points), use_container_width=True, hide_index=True)

    if scenarios:
        st.subheader("Scenario Risk Summary")
        for s in scenarios:
            cA, cB, cC = st.columns([1.2, 1, 4])
            cA.markdown(f"**{s['scenario']}**")
            cB.markdown(risk_badge(s.get("risk_level", "unknown")), unsafe_allow_html=True)
            cC.markdown(
                f"`{float(s.get('flood_ratio', 0.0))*100:.2f}%` flooded | "
                f"`{float(s.get('flood_area_m2', 0.0)):,.0f} m2` | "
                f"`{int(s.get('component_count', 0))}` coastal components"
            )

    with st.expander("Raw JSON"):
        st.json(payload)


def render_compare(payloads: list[dict], selected_scenario: str):
    st.subheader("Multi-city Compare")
    summary_rows = []
    chart_rows = []

    for payload in payloads:
        city_key = payload.get("city") or payload.get("display_name")
        scenario = next((s for s in payload.get("scenarios", []) if s["scenario"] == selected_scenario), None)
        summary_rows.append(
            {
                "city": payload.get("display_name") or city_key,
                "provider": payload.get("provider_label"),
                "support_tier": payload.get("support_tier"),
                "station": payload.get("station"),
                "forecast_mode": payload.get("model", {}).get("forecast_mode_used"),
                "peak_prediction_m": round(float(payload.get("peak_prediction_m", 0.0)), 4),
                "last_observation_utc": payload.get("history", {}).get("last_observation_utc"),
                "observation_delay_hours": None if payload.get("source", {}).get("observation_delay_hours") is None else round(float(payload["source"]["observation_delay_hours"]), 1),
                f"{selected_scenario}_flood_ratio_pct": None if scenario is None else round(float(scenario.get("flood_ratio", 0.0)) * 100.0, 3),
                f"{selected_scenario}_flood_area_m2": None if scenario is None else round(float(scenario.get("flood_area_m2", 0.0)), 1),
                f"{selected_scenario}_risk": None if scenario is None else scenario.get("risk_level"),
            }
        )

        for point in payload.get("forecast", []):
            chart_rows.append(
                {
                    "city": payload.get("display_name") or city_key,
                    "hour_ahead": point["hour_ahead"],
                    "sea_level_m": point["sea_level_m"],
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    if chart_rows:
        df_chart = pd.DataFrame(chart_rows)
        pivot = df_chart.pivot(index="hour_ahead", columns="city", values="sea_level_m")
        st.subheader("Forecast Compare")
        st.line_chart(pivot)

    st.subheader(f"{selected_scenario} Risk Cards")
    cols = st.columns(min(4, max(1, len(payloads))))
    for idx, payload in enumerate(payloads):
        scenario = next((s for s in payload.get("scenarios", []) if s["scenario"] == selected_scenario), None)
        with cols[idx % len(cols)]:
            st.markdown(f"**{payload.get('display_name') or payload.get('city')}**")
            st.caption(f"{payload.get('provider_label')} | {payload.get('support_tier')}")
            st.metric("Peak (m)", f"{float(payload.get('peak_prediction_m', 0.0)):.3f}")
            if scenario:
                st.markdown(risk_badge(scenario.get("risk_level", "unknown")), unsafe_allow_html=True)
                st.caption(
                    f"{selected_scenario}: {float(scenario.get('flood_ratio', 0.0))*100:.2f}% | "
                    f"{float(scenario.get('flood_area_m2', 0.0)):,.0f} m2"
                )
            if payload.get("source", {}).get("note"):
                st.caption(payload["source"]["note"])

    st.subheader("Operational Briefs")
    for payload in payloads:
        summary = selected_operational_summary(payload, selected_scenario)
        if not summary:
            continue
        with st.expander(f"{payload.get('display_name') or payload.get('city')} | {str(summary.get('alert_level', 'n/a')).upper()}"):
            st.markdown(f"**{summary.get('headline', 'No headline available.')}**")
            st.caption(summary.get("summary_text", ""))
            for item in summary.get("recommended_actions", []):
                st.markdown(f"- {item}")
            impact = selected_impact_summary(payload, selected_scenario)
            if impact and impact.get("top_hotspots"):
                st.markdown("**Top Hotspots**")
                for hotspot in impact["top_hotspots"][:3]:
                    st.markdown(
                        f"- Rank {int(hotspot['rank'])}: {hotspot['risk_level']} | "
                        f"score {float(hotspot['priority_score']):.1f} | "
                        f"{float(hotspot['area_m2']):,.0f} m2"
                    )


st.set_page_config(page_title="Coastal Flood Risk", layout="wide")
st.title("Coastal Flood Risk Dashboard")

registry = load_city_registry()
all_city_keys = sorted(registry.keys(), key=lambda k: registry[k].get("display_name", k))
full_map_ready_cities, partial_map_cities, forecast_only_cities = split_city_keys_by_map_status(
    all_city_keys,
    outputs_root=REPO_ROOT / "Backend" / "sea_level_risk" / "outputs" / "realtime",
)
city_keys = full_map_ready_cities or all_city_keys

def _city_labels(keys: list[str]) -> str:
    return ", ".join(registry[key].get("display_name", key) for key in keys)

with st.sidebar:
    st.header("Controls")
    st.caption("Demo-safe mode: only cities with full local flood-map assets are shown.")
    if partial_map_cities:
        st.caption(f"Hidden partial-map cities: {_city_labels(partial_map_cities)}")
    if forecast_only_cities:
        st.caption(f"Hidden forecast-only cities: {_city_labels(forecast_only_cities)}")
    api_base = st.text_input("Realtime API URL", "http://127.0.0.1:8000")
    dashboard_mode = st.selectbox("Dashboard mode", ["Single city", "Multi-city compare"], index=0)
    default_index = city_keys.index("honolulu") if "honolulu" in city_keys else 0
    city = st.selectbox("City", city_keys, index=default_index, format_func=lambda key: registry[key].get("display_name", key))
    compare_defaults = [c for c in ["honolulu", "boston", "newyork", "jakarta"] if c in city_keys]
    compare_cities = st.multiselect(
        "Compare cities",
        city_keys,
        default=compare_defaults,
        format_func=lambda key: registry[key].get("display_name", key),
        disabled=dashboard_mode != "Multi-city compare",
    )
    horizon = st.slider("Forecast horizon (hours)", 1, 24, 6)
    hours_back = st.slider("History window (hours)", 48, 240, 96, step=24)
    scenario = st.selectbox("Scenario", ["plus_20cm", "plus_50cm", "plus_100cm"], index=1)
    show_all = st.checkbox("Overlay all scenarios", value=True, disabled=dashboard_mode != "Single city")
    map_mode = st.selectbox("Map mode", ["2D flood map", "3D terrain map", "Both"], index=0, disabled=dashboard_mode != "Single city")
    camera_preset = st.selectbox("Camera", ["oblique", "top", "coastal"], index=0, disabled=dashboard_mode != "Single city")
    downsample = st.slider("3D downsample", 2, 10, 4, disabled=dashboard_mode != "Single city")
    zex = st.slider("Vertical exaggeration", 1.0, 8.0, 2.0, 0.5, disabled=dashboard_mode != "Single city")
    run = st.button("Refresh Realtime")

if run:
    try:
        if dashboard_mode == "Single city":
            payload = fetch_payload(api_base=api_base, city=city, horizon=horizon, hours_back=hours_back, auto_dem=True)
            render_single_city(
                payload=payload,
                api_base=api_base,
                city=city,
                horizon=horizon,
                hours_back=hours_back,
                scenario=scenario,
                show_all=show_all,
                map_mode=map_mode,
                camera_preset=camera_preset,
                downsample=downsample,
                zex=zex,
            )
        else:
            selected = compare_cities or compare_defaults or city_keys[: min(4, len(city_keys))]
            payloads = [fetch_payload(api_base=api_base, city=c, horizon=horizon, hours_back=hours_back, auto_dem=True) for c in selected]
            render_compare(payloads=payloads, selected_scenario=scenario)
    except Exception as exc:
        st.error(f"API error: {exc}")
else:
    st.info("Select controls and click 'Refresh Realtime'.")

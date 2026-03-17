from pathlib import Path

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from .city_registry import load_city_registry
    from .render_2d import SCENARIO_2D_STYLE, build_2d_layers
    from .render_3d import render_3d_flood_map_multi
except ImportError:
    from Backend.sea_level_risk.city_registry import load_city_registry
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


def render_single_city(payload: dict, api_base: str, city: str, horizon: int, hours_back: int, scenario: str, show_all: bool, map_mode: str, camera_preset: str, downsample: int, zex: float):
    render_header(payload)

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


st.set_page_config(page_title="Coastal Water-Level Realtime 3D", layout="wide")
st.title("Realtime Coastal Water-Level and Flood Risk Dashboard (3D GIS)")

registry = load_city_registry()
city_keys = sorted(registry.keys(), key=lambda k: registry[k].get("display_name", k))

with st.sidebar:
    st.header("Controls")
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

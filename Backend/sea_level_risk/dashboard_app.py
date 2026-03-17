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


st.set_page_config(page_title="Coastal Water-Level Realtime 3D", layout="wide")
st.title("Realtime Coastal Water-Level and Flood Risk Dashboard (3D GIS)")

registry = load_city_registry()
city_keys = sorted(registry.keys(), key=lambda k: registry[k].get("display_name", k))

with st.sidebar:
    st.header("Controls")
    api_base = st.text_input("Realtime API URL", "http://127.0.0.1:8000")
    default_index = city_keys.index("honolulu") if "honolulu" in city_keys else 0
    city = st.selectbox("City", city_keys, index=default_index, format_func=lambda key: registry[key].get("display_name", key))
    horizon = st.slider("Forecast horizon (hours)", 1, 24, 6)
    hours_back = st.slider("History window (hours)", 48, 240, 96, step=24)
    scenario = st.selectbox("Scenario", ["plus_20cm", "plus_50cm", "plus_100cm"], index=1)
    show_all = st.checkbox("Overlay all scenarios", value=True)
    map_mode = st.selectbox("Map mode", ["2D flood map", "3D terrain map", "Both"], index=0)
    camera_preset = st.selectbox("Camera", ["oblique", "top", "coastal"], index=0)
    downsample = st.slider("3D downsample", 2, 10, 4)
    zex = st.slider("Vertical exaggeration", 1.0, 8.0, 2.0, 0.5)
    run = st.button("Refresh Realtime")

if run:
    params = {
        "city": city,
        "horizon": horizon,
        "hours_back": hours_back,
        "datum": "MSL",
        "auto_dem": 1,
    }
    url = f"{api_base.rstrip('/')}/realtime/forecast"

    try:
        resp = requests.get(url, params=params, timeout=180)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        st.stop()

    if "error" in payload:
        st.error(payload["error"])
        st.stop()

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

    scenarios = payload.get("scenarios", [])
    scenario_map = {s["scenario"]: s for s in scenarios}
    hotspot_geojson = Path("Backend/sea_level_risk/outputs/realtime") / city / "hotspots.geojson"
    if scenarios and not hotspot_geojson.exists():
        try:
            requests.get(
                f"{api_base.rstrip('/')}/realtime/hotspots",
                params={"city": city, "limit": 20, "horizon": horizon, "hours_back": hours_back, "datum": "MSL"},
                timeout=180,
            )
        except Exception:
            pass

    if show_all:
        items = [
            {
                "scenario": s["scenario"],
                "flood_geojson": s["geojson"],
                "water_level_m": s.get("scenario_water_level_m"),
            }
            for s in scenarios
        ]
        map_name = "map3d_all_scenarios.html"
    else:
        s_obj = scenario_map.get(scenario)
        items = [] if s_obj is None else [{
            "scenario": s_obj["scenario"],
            "flood_geojson": s_obj["geojson"],
            "water_level_m": s_obj.get("scenario_water_level_m"),
        }]
        map_name = f"map3d_{scenario}.html"

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
    if dem_path and Path(dem_path).exists() and scenarios and map_mode in {"3D terrain map", "Both"}:
        if items:
            st.subheader("3D Terrain Map")
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
else:
    st.info("Select controls and click 'Refresh Realtime'.")

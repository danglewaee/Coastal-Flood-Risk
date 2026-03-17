from pathlib import Path

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

try:
    from .city_registry import load_city_registry
    from .render_2d import SCENARIO_2D_STYLE, build_2d_layers
except ImportError:
    from Backend.sea_level_risk.city_registry import load_city_registry
    from Backend.sea_level_risk.render_2d import SCENARIO_2D_STYLE, build_2d_layers


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


def badge(text: str, tone: str = "neutral") -> str:
    tones = {
        "neutral": ("rgba(255,255,255,0.08)", "#f7f5ef"),
        "good": ("rgba(51,195,143,0.18)", "#6ff0bf"),
        "warn": ("rgba(255,184,77,0.18)", "#ffd27f"),
        "bad": ("rgba(247,92,92,0.18)", "#ff9b9b"),
        "info": ("rgba(97,175,255,0.18)", "#90c2ff"),
    }
    bg, fg = tones.get(tone, tones["neutral"])
    return f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;background:{bg};color:{fg};font-size:12px;font-weight:700;letter-spacing:0.02em'>{text}</span>"


def support_tone(value: str) -> str:
    if value == "official_realtime":
        return "good"
    if value == "experimental_realtime":
        return "warn"
    if value == "proxy_delayed":
        return "bad"
    return "neutral"


def scenario_summary_row(payload: dict, selected_scenario: str) -> dict:
    scenario = next((s for s in payload.get("scenarios", []) if s["scenario"] == selected_scenario), None)
    return {
        "city": payload.get("display_name") or payload.get("city"),
        "provider": payload.get("provider_label"),
        "support_tier": payload.get("support_tier"),
        "station": payload.get("station"),
        "peak_prediction_m": round(float(payload.get("peak_prediction_m", 0.0)), 4),
        "delay_h": None if payload.get("source", {}).get("observation_delay_hours") is None else round(float(payload["source"]["observation_delay_hours"]), 1),
        "scenario": selected_scenario,
        "flood_ratio_pct": None if scenario is None else round(float(scenario.get("flood_ratio", 0.0)) * 100.0, 3),
        "flood_area_m2": None if scenario is None else round(float(scenario.get("flood_area_m2", 0.0)), 1),
        "components": None if scenario is None else int(scenario.get("component_count", 0)),
        "risk_level": None if scenario is None else scenario.get("risk_level"),
    }


st.set_page_config(page_title="Sea-Level Risk Presentation", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(49,125,255,0.12), transparent 30%),
            radial-gradient(circle at top right, rgba(14,150,122,0.12), transparent 26%),
            linear-gradient(180deg, #0b1220 0%, #101826 40%, #121926 100%);
        color: #f5f3ee;
    }
    div[data-testid="stMetricValue"] {
        color: #f8f5ef;
    }
    div[data-testid="stMetricLabel"] {
        color: #9db0c7;
    }
    .hero {
        padding: 28px 30px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(13,71,161,0.55), rgba(17,84,77,0.42));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 60px rgba(0,0,0,0.22);
        margin-bottom: 18px;
    }
    .hero h1 {
        margin: 0 0 10px 0;
        font-size: 3rem;
        line-height: 1.05;
        color: #f8f5ef;
    }
    .hero p {
        margin: 0;
        color: #c8d6e6;
        max-width: 900px;
        font-size: 1.05rem;
    }
    .glass {
        padding: 18px 18px 16px 18px;
        border-radius: 20px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Realtime Coastal Water-Level And Flood Risk</h1>
      <p>
        Presentation mode for a multi-city prototype: official NOAA gauges where possible,
        experimental IOC feeds where necessary, short-horizon forecasts, and DEM-based coastal flood scenarios.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

registry = load_city_registry()
city_keys = sorted(registry.keys(), key=lambda k: registry[k].get("display_name", k))

with st.sidebar:
    st.header("Presentation Controls")
    api_base = st.text_input("Realtime API URL", "http://127.0.0.1:8100")
    spotlight_city = st.selectbox("Spotlight city", city_keys, index=city_keys.index("boston") if "boston" in city_keys else 0, format_func=lambda key: registry[key].get("display_name", key))
    compare_default = [c for c in ["boston", "newyork", "jakarta", "amsterdam"] if c in city_keys]
    compare_cities = st.multiselect("Compare cities", city_keys, default=compare_default, format_func=lambda key: registry[key].get("display_name", key))
    scenario = st.selectbox("Scenario", ["plus_20cm", "plus_50cm", "plus_100cm"], index=1)
    show_all = st.checkbox("Overlay all scenarios on spotlight map", value=False)
    horizon = st.slider("Forecast horizon (hours)", 1, 24, 6)
    hours_back = st.slider("History window (hours)", 48, 240, 96, step=24)
    run = st.button("Build Presentation View")

if run:
    try:
        spotlight = fetch_payload(api_base=api_base, city=spotlight_city, horizon=horizon, hours_back=hours_back, auto_dem=True)
        compare_selection = compare_cities or compare_default or city_keys[: min(4, len(city_keys))]
        compare_payloads = [fetch_payload(api_base=api_base, city=city, horizon=horizon, hours_back=hours_back, auto_dem=True) for city in compare_selection]
    except Exception as exc:
        st.error(f"Presentation data fetch failed: {exc}")
        st.stop()

    spotlight_scenario = next((s for s in spotlight.get("scenarios", []) if s["scenario"] == scenario), None)

    top_left, top_right = st.columns([1.25, 1.0], gap="large")

    with top_left:
        st.markdown("### City Spotlight")
        st.markdown(
            f"{badge(spotlight.get('provider_label', 'n/a'), 'info')} "
            f"{badge(spotlight.get('support_tier', 'n/a'), support_tone(spotlight.get('support_tier', 'n/a')))} "
            f"{badge(spotlight.get('model', {}).get('forecast_mode_used', 'n/a'), 'neutral')}",
            unsafe_allow_html=True,
        )
        st.subheader(spotlight.get("display_name") or spotlight.get("city"))
        if spotlight.get("city_notes"):
            st.caption(spotlight["city_notes"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak Forecast (m)", f"{float(spotlight.get('peak_prediction_m', 0.0)):.3f}")
        m2.metric("Station", str(spotlight.get("station", "n/a")))
        m3.metric("Obs Delay (h)", "n/a" if spotlight.get("source", {}).get("observation_delay_hours") is None else f"{float(spotlight['source']['observation_delay_hours']):.1f}")
        m4.metric(
            f"{scenario} Flood %",
            "n/a" if spotlight_scenario is None else f"{float(spotlight_scenario.get('flood_ratio', 0.0))*100:.2f}%",
        )

        forecast_points = spotlight.get("forecast", [])
        if forecast_points:
            st.markdown("#### Forecast Trajectory")
            forecast_df = pd.DataFrame(forecast_points)
            st.line_chart(forecast_df, x="hour_ahead", y="sea_level_m")

        if spotlight_scenario:
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{scenario} Flood Area", f"{float(spotlight_scenario.get('flood_area_m2', 0.0)):,.0f} m2")
            c2.metric(f"{scenario} Components", f"{int(spotlight_scenario.get('component_count', 0))}")
            c3.metric(f"{scenario} Risk", str(spotlight_scenario.get("risk_level", "n/a")).upper())

    with top_right:
        st.markdown("### Spotlight Flood Map")
        spotlight_items = scenario_items_from_payload(spotlight, selected_scenario=scenario, show_all=show_all)
        hotspot_geojson = Path("Backend/sea_level_risk/outputs/realtime") / spotlight_city / "hotspots.geojson"
        layers, view_state = build_2d_layers(
            spotlight_items,
            hotspot_geojson=str(hotspot_geojson) if hotspot_geojson.exists() else None,
        )
        if layers and view_state:
            deck = pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=view_state,
                layers=layers,
                tooltip={
                    "html": "<b>{scenario_label}</b><br/>Risk: {risk_level}<br/>Priority: {priority_score}",
                    "style": {"backgroundColor": "#10243a", "color": "white"},
                },
            )
            st.pydeck_chart(deck, use_container_width=True)
            legend = []
            for key in ["plus_20cm", "plus_50cm", "plus_100cm"]:
                style = SCENARIO_2D_STYLE[key]
                legend.append(
                    f"<span style='display:inline-block;width:12px;height:12px;background:rgba({style['rgba'][0]},{style['rgba'][1]},{style['rgba'][2]},0.8);margin-right:6px;border-radius:2px;'></span>{style['label']}"
                )
            st.markdown(" | ".join(legend), unsafe_allow_html=True)
        else:
            st.warning("No spotlight map could be rendered for the selected city/scenario.")

    st.markdown("---")
    st.markdown("### Cross-city Compare")

    compare_rows = [scenario_summary_row(payload, scenario) for payload in compare_payloads]
    compare_df = pd.DataFrame(compare_rows).sort_values(["support_tier", "flood_ratio_pct", "peak_prediction_m"], ascending=[True, False, False], na_position="last")

    t1, t2 = st.columns([1.4, 1.0], gap="large")
    with t1:
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

    with t2:
        risk_df = compare_df[["city", "flood_ratio_pct", "flood_area_m2"]].set_index("city")
        if not risk_df.empty:
            st.markdown("#### Scenario Compare")
            st.bar_chart(risk_df)

    st.markdown("### Executive Takeaways")
    takeaway_cols = st.columns(min(4, max(1, len(compare_rows))))
    for idx, row in enumerate(compare_rows[: len(takeaway_cols)]):
        tone = support_tone(row["support_tier"])
        with takeaway_cols[idx]:
            st.markdown(f"<div class='glass'><strong>{row['city']}</strong><br/>{badge(row['support_tier'], tone)}<br/><br/>Peak forecast: <strong>{row['peak_prediction_m']:.3f} m</strong><br/>{scenario}: <strong>{0.0 if row['flood_ratio_pct'] is None else row['flood_ratio_pct']:.2f}%</strong><br/>Delay: <strong>{'n/a' if row['delay_h'] is None else str(row['delay_h']) + ' h'}</strong></div>", unsafe_allow_html=True)

    st.markdown("### Method And Limits")
    st.markdown(
        """
        - Realtime water level comes from official NOAA gauges where available, and IOC public feeds where NOAA is unavailable.
        - Forecast is deep-learning only for Honolulu at the moment; other cities use a tide-aware short-horizon baseline.
        - Flood polygons are GIS threshold outputs on Copernicus DEM, filtered to coast-connected components.
        - `Amsterdam` is shown as a delayed regional proxy. It should not be presented as a direct Amsterdam city gauge.
        """
    )
else:
    st.info("Configure the spotlight city and compare set, then click `Build Presentation View`.")

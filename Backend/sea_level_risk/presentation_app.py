from pathlib import Path
import sys

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .asset_readiness import split_city_keys_by_map_status
    from .city_registry import load_city_registry
    from .operational_summary import build_briefing_markdown
    from .render_2d import SCENARIO_2D_STYLE, build_2d_layers
except ImportError:
    from Backend.sea_level_risk.asset_readiness import split_city_keys_by_map_status
    from Backend.sea_level_risk.city_registry import load_city_registry
    from Backend.sea_level_risk.operational_summary import build_briefing_markdown
    from Backend.sea_level_risk.render_2d import SCENARIO_2D_STYLE, build_2d_layers


def fetch_payload(
    api_base: str,
    city: str,
    horizon: int,
    hours_back: int,
    auto_dem: bool = True,
    scenario_names: list[str] | None = None,
) -> dict:
    url = f"{api_base.rstrip('/')}/realtime/forecast"
    params = {
        "city": city,
        "horizon": horizon,
        "hours_back": hours_back,
        "datum": "MSL",
        "auto_dem": 1 if auto_dem else 0,
    }
    if scenario_names:
        params["scenarios"] = ",".join(scenario_names)
    resp = requests.get(url, params=params, timeout=180)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


@st.cache_data(ttl=120, show_spinner=False)
def fetch_payload_cached(
    api_base: str,
    city: str,
    horizon: int,
    hours_back: int,
    auto_dem: bool = True,
    scenario_names: tuple[str, ...] | None = None,
) -> dict:
    return fetch_payload(
        api_base=api_base,
        city=city,
        horizon=horizon,
        hours_back=hours_back,
        auto_dem=auto_dem,
        scenario_names=list(scenario_names) if scenario_names else None,
    )


def load_presentation_data(
    api_base: str,
    spotlight_city: str,
    compare_selection: list[str],
    horizon: int,
    hours_back: int,
    auto_dem: bool,
    selected_scenario: str,
    show_all: bool,
) -> tuple[dict, list[dict], list[str]]:
    spotlight_scenarios = tuple(SCENARIO_2D_STYLE.keys()) if show_all else (selected_scenario,)
    spotlight = fetch_payload_cached(
        api_base=api_base,
        city=spotlight_city,
        horizon=horizon,
        hours_back=hours_back,
        auto_dem=auto_dem,
        scenario_names=spotlight_scenarios,
    )

    compare_payloads: list[dict] = []
    warnings: list[str] = []
    for city in compare_selection:
        try:
            compare_payloads.append(
                fetch_payload_cached(
                    api_base=api_base,
                    city=city,
                    horizon=horizon,
                    hours_back=hours_back,
                    auto_dem=auto_dem,
                    scenario_names=(selected_scenario,),
                )
            )
        except Exception as exc:
            warnings.append(f"{registry.get(city, {}).get('display_name', city)}: {exc}")

    return spotlight, compare_payloads, warnings


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


def alert_tone(value: str) -> str:
    if value == "critical":
        return "bad"
    if value == "high":
        return "warn"
    if value == "moderate":
        return "info"
    if value == "low":
        return "good"
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


st.set_page_config(page_title="Coastal Flood Risk", layout="wide")

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
        padding: 28px 30px 22px 30px;
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
      <h1>Coastal Flood Risk</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

registry = load_city_registry()
all_city_keys = sorted(registry.keys(), key=lambda k: registry[k].get("display_name", k))
full_map_ready_cities, partial_map_cities, forecast_only_cities = split_city_keys_by_map_status(
    all_city_keys,
    outputs_root=REPO_ROOT / "Backend" / "sea_level_risk" / "outputs" / "realtime",
)
city_keys = full_map_ready_cities or all_city_keys
compare_default = [c for c in ["boston", "newyork", "jakarta", "amsterdam"] if c in city_keys]

def _city_labels(keys: list[str]) -> str:
    return ", ".join(registry[key].get("display_name", key) for key in keys)

default_view = {
    "api_base": "http://127.0.0.1:8100",
    "spotlight_city": "boston" if "boston" in city_keys else city_keys[0],
    "compare_cities": compare_default,
    "scenario": "plus_50cm",
    "show_all": False,
    "horizon": 6,
    "hours_back": 96,
    "auto_dem": False,
}

if "presentation_loaded" not in st.session_state:
    st.session_state.presentation_loaded = False
if "presentation_view" not in st.session_state:
    st.session_state.presentation_view = default_view.copy()
if "presentation_spotlight" not in st.session_state:
    st.session_state.presentation_spotlight = None
if "presentation_compare_payloads" not in st.session_state:
    st.session_state.presentation_compare_payloads = []
if "presentation_fetch_warnings" not in st.session_state:
    st.session_state.presentation_fetch_warnings = []

with st.sidebar:
    st.header("Presentation Controls")
    st.caption("Demo-safe mode: only cities with full local flood-map assets are shown.")
    if partial_map_cities:
        st.caption(f"Hidden partial-map cities: {_city_labels(partial_map_cities)}")
    if forecast_only_cities:
        st.caption(f"Hidden forecast-only cities: {_city_labels(forecast_only_cities)}")
    with st.form("presentation_controls"):
        current_spotlight = st.session_state.presentation_view["spotlight_city"]
        if current_spotlight not in city_keys:
            current_spotlight = city_keys[0]
        current_compare = [key for key in st.session_state.presentation_view["compare_cities"] if key in city_keys]
        if not current_compare:
            current_compare = compare_default or city_keys[: min(4, len(city_keys))]
        api_base = st.text_input("Realtime API URL", st.session_state.presentation_view["api_base"])
        spotlight_city = st.selectbox(
            "Spotlight city",
            city_keys,
            index=city_keys.index(current_spotlight),
            format_func=lambda key: registry[key].get("display_name", key),
        )
        compare_cities = st.multiselect(
            "Compare cities",
            city_keys,
            default=current_compare,
            format_func=lambda key: registry[key].get("display_name", key),
        )
        scenario = st.selectbox(
            "Scenario",
            ["plus_20cm", "plus_50cm", "plus_100cm"],
            index=["plus_20cm", "plus_50cm", "plus_100cm"].index(st.session_state.presentation_view["scenario"]),
        )
        show_all = st.checkbox("Overlay all scenarios on spotlight map", value=st.session_state.presentation_view["show_all"])
        auto_dem = st.checkbox(
            "Allow auto-download DEM for missing cities",
            value=st.session_state.presentation_view["auto_dem"],
            help="Leave this off for stable switching. Turn it on only when preparing a city for the first time.",
        )
        horizon = st.slider("Forecast horizon (hours)", 1, 24, st.session_state.presentation_view["horizon"])
        hours_back = st.slider("History window (hours)", 48, 240, st.session_state.presentation_view["hours_back"], step=24)
        refresh = st.form_submit_button("Refresh Presentation View")

requested_view = {
    "api_base": api_base,
    "spotlight_city": spotlight_city,
    "compare_cities": compare_cities or compare_default or city_keys[: min(4, len(city_keys))],
    "scenario": scenario,
    "show_all": show_all,
    "horizon": horizon,
    "hours_back": hours_back,
    "auto_dem": auto_dem,
}

if refresh:
    fetch_payload_cached.clear()

if refresh or not st.session_state.presentation_loaded:
    try:
        with st.spinner("Building presentation view..."):
            spotlight, compare_payloads, fetch_warnings = load_presentation_data(
                api_base=requested_view["api_base"],
                spotlight_city=requested_view["spotlight_city"],
                compare_selection=requested_view["compare_cities"],
                horizon=requested_view["horizon"],
                hours_back=requested_view["hours_back"],
                auto_dem=requested_view["auto_dem"],
                selected_scenario=requested_view["scenario"],
                show_all=requested_view["show_all"],
            )
        st.session_state.presentation_spotlight = spotlight
        st.session_state.presentation_compare_payloads = compare_payloads
        st.session_state.presentation_fetch_warnings = fetch_warnings
        st.session_state.presentation_view = requested_view
        st.session_state.presentation_loaded = True
    except Exception as exc:
        st.error(f"Presentation data fetch failed: {exc}")
        if not st.session_state.presentation_loaded:
            st.stop()

loaded_view = st.session_state.presentation_view
spotlight = st.session_state.presentation_spotlight
compare_payloads = st.session_state.presentation_compare_payloads
fetch_warnings = st.session_state.presentation_fetch_warnings

if requested_view != loaded_view:
    st.info("Controls changed but current view is still showing the last loaded data. Click `Refresh Presentation View` to apply the new city/settings.")

for warning in fetch_warnings:
    st.warning(f"Compare city skipped: {warning}")

if not spotlight:
    st.error("No presentation data is currently loaded.")
    st.stop()

spotlight_scenario = next((s for s in spotlight.get("scenarios", []) if s["scenario"] == loaded_view["scenario"]), None)

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
        f"{loaded_view['scenario']} Flood %",
        "n/a" if spotlight_scenario is None else f"{float(spotlight_scenario.get('flood_ratio', 0.0))*100:.2f}%",
    )

    spotlight_summary = selected_operational_summary(spotlight, loaded_view["scenario"])
    if spotlight_summary:
        st.markdown("#### Operational Summary")
        st.markdown(
            f"{badge(str(spotlight_summary.get('alert_level', 'n/a')).upper(), alert_tone(spotlight_summary.get('alert_level', 'n/a')))} "
            f"{badge(str(spotlight_summary.get('confidence', 'n/a')).upper(), 'neutral')}",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{spotlight_summary.get('headline', 'No operational headline available.')}**")
        st.caption(spotlight_summary.get("summary_text", ""))
        for item in spotlight_summary.get("recommended_actions", []):
            st.markdown(f"- {item}")
        st.caption(spotlight_summary.get("thresholds_note", ""))

        briefing_md = build_briefing_markdown(
            spotlight,
            city_cfg=registry.get(loaded_view["spotlight_city"], {}),
            scenario_name=spotlight_summary.get("scenario_basis"),
        )
        st.download_button(
            "Download Operational Briefing",
            data=briefing_md,
            file_name=f"{loaded_view['spotlight_city']}_{spotlight_summary.get('scenario_basis') or 'briefing'}_briefing.md",
            mime="text/markdown",
        )

    spotlight_impact = selected_impact_summary(spotlight, loaded_view["scenario"])
    if spotlight_impact:
        st.markdown("#### Priority Hotspots")
        c1, c2, c3 = st.columns(3)
        c1.metric("Hotspots Ranked", f"{int(spotlight_impact.get('hotspot_count', 0))}")
        c2.metric(
            "Largest Hotspot",
            "n/a" if spotlight_impact.get("largest_hotspot_area_m2") is None else f"{float(spotlight_impact['largest_hotspot_area_m2']):,.0f} m2",
        )
        c3.metric("Exposure Layers", f"{int(spotlight_impact.get('exposure_layers_available', 0))}")
        for hotspot in spotlight_impact.get("top_hotspots", [])[:3]:
            st.markdown(
                f"- **Rank {int(hotspot['rank'])}** | {str(hotspot['risk_level']).upper()} | "
                f"score {float(hotspot['priority_score']):.1f} | "
                f"{float(hotspot['area_m2']):,.0f} m2"
            )
        rollup_rows = spotlight_impact.get("exposure_rollup", [])
        if rollup_rows:
            st.markdown("#### Operational Impact")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric(
                "Road Length Affected",
                "n/a" if spotlight_impact.get("affected_road_length_m") is None else f"{float(spotlight_impact['affected_road_length_m'])/1000.0:.2f} km",
            )
            e2.metric("Priority Sites Affected", f"{int(spotlight_impact.get('affected_site_count_total', 0))}")
            e3.metric("Impact Categories", f"{int(spotlight_impact.get('categories_impacted', 0))}")
            e4.metric(
                "Population Affected",
                "n/a" if spotlight_impact.get("population_affected_estimate") in {None, 0} else f"{int(spotlight_impact['population_affected_estimate']):,}",
            )
            for item in spotlight_impact.get("impact_headline_items", []):
                st.markdown(f"- {item}")
            rollup_df = pd.DataFrame(rollup_rows).copy()
            visible_rollup_df = rollup_df[rollup_df["affected_value"].astype(float) > 0].copy()
            if visible_rollup_df.empty:
                visible_rollup_df = rollup_df
            st.dataframe(
                visible_rollup_df[["display_name", "affected_value", "affected_unit"]],
                use_container_width=True,
                hide_index=True,
            )
        exposure_rows = spotlight_impact.get("exposure_summary", [])
        if exposure_rows:
            st.markdown("#### Exposure Detail")
            exposure_df = pd.DataFrame(exposure_rows).copy()
            for col in ["affected_area_m2", "affected_length_m"]:
                if col in exposure_df.columns:
                    exposure_df[col] = exposure_df[col].map(lambda v: round(float(v), 1))
            if "affected_value" in exposure_df.columns:
                exposure_df["affected_value"] = exposure_df["affected_value"].map(lambda v: round(float(v), 1) if isinstance(v, float) else v)
            st.dataframe(exposure_df, use_container_width=True, hide_index=True)

    forecast_points = spotlight.get("forecast", [])
    if forecast_points:
        st.markdown("#### Forecast Trajectory")
        forecast_df = pd.DataFrame(forecast_points)
        if {"p10_m", "p50_m", "p90_m"}.issubset(forecast_df.columns):
            st.line_chart(forecast_df, x="hour_ahead", y=["p10_m", "p50_m", "p90_m"])
            if spotlight.get("model", {}).get("uncertainty", {}).get("method"):
                st.caption(f"Uncertainty method: {spotlight['model']['uncertainty']['method']}")
        else:
            st.line_chart(forecast_df, x="hour_ahead", y="sea_level_m")

    if spotlight_scenario:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{loaded_view['scenario']} Flood Area", f"{float(spotlight_scenario.get('flood_area_m2', 0.0)):,.0f} m2")
        c2.metric(f"{loaded_view['scenario']} Components", f"{int(spotlight_scenario.get('component_count', 0))}")
        c3.metric(f"{loaded_view['scenario']} Risk", str(spotlight_scenario.get("risk_level", "n/a")).upper())

with top_right:
    st.markdown("### Spotlight Flood Map")
    spotlight_items = scenario_items_from_payload(spotlight, selected_scenario=loaded_view["scenario"], show_all=loaded_view["show_all"])
    hotspot_geojson = Path("Backend/sea_level_risk/outputs/realtime") / loaded_view["spotlight_city"] / "hotspots.geojson"
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

compare_rows = [scenario_summary_row(payload, loaded_view["scenario"]) for payload in compare_payloads]
if compare_rows:
    compare_df = pd.DataFrame(compare_rows).sort_values(["support_tier", "flood_ratio_pct", "peak_prediction_m"], ascending=[True, False, False], na_position="last")

    t1, t2 = st.columns([1.4, 1.0], gap="large")
    with t1:
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

    with t2:
        risk_df = compare_df[["city", "flood_ratio_pct", "flood_area_m2"]].set_index("city")
        if not risk_df.empty:
            st.markdown("#### Scenario Compare")
            st.bar_chart(risk_df)
else:
    st.warning("No compare-city payloads are currently available.")

st.markdown("### Executive Takeaways")
takeaway_cols = st.columns(min(4, max(1, len(compare_rows))))
for idx, row in enumerate(compare_rows[: len(takeaway_cols)]):
    tone = support_tone(row["support_tier"])
    with takeaway_cols[idx]:
        st.markdown(f"<div class='glass'><strong>{row['city']}</strong><br/>{badge(row['support_tier'], tone)}<br/><br/>Peak forecast: <strong>{row['peak_prediction_m']:.3f} m</strong><br/>{loaded_view['scenario']}: <strong>{0.0 if row['flood_ratio_pct'] is None else row['flood_ratio_pct']:.2f}%</strong><br/>Delay: <strong>{'n/a' if row['delay_h'] is None else str(row['delay_h']) + ' h'}</strong></div>", unsafe_allow_html=True)

st.markdown("### Method And Limits")
st.markdown(
    """
    - Realtime water level comes from official NOAA gauges where available, and IOC public feeds where NOAA is unavailable.
    - Forecast uses city-specific deep-learning models with multivariate derived features and calibrated uncertainty bands where hourly training histories are available; lower-data cities fall back to a tide-aware short-horizon baseline.
    - Flood polygons are GIS threshold outputs on Copernicus DEM, filtered to coast-connected components.
    - `Amsterdam` is shown as a delayed regional proxy. It should not be presented as a direct Amsterdam city gauge.
    """
)

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

    return spotlight, [], []


def load_compare_data(
    api_base: str,
    compare_selection: list[str],
    horizon: int,
    hours_back: int,
    auto_dem: bool,
    selected_scenario: str,
) -> tuple[list[dict], list[str]]:
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

    return compare_payloads, warnings


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

    st.markdown("#### Impacted Site Types")
    affected_rows = [row for row in site_rows if row["affected_sites"] > 0]
    if affected_rows:
        cols = st.columns(min(4, len(affected_rows)))
        for idx, row in enumerate(affected_rows[:4]):
            cols[idx % len(cols)].metric(row["site_type"], f"{row['affected_sites']:,}")

    st.dataframe(pd.DataFrame(site_rows), use_container_width=True, hide_index=True)


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
            default=[current_spotlight],
            format_func=lambda key: registry[key].get("display_name", key),
            key="compare_cities_selector_v2",
        )
        scenario = st.selectbox(
            "Scenario",
            ["plus_20cm", "plus_50cm", "plus_100cm"],
            index=["plus_20cm", "plus_50cm", "plus_100cm"].index(st.session_state.presentation_view["scenario"]),
        )
        show_all = st.checkbox("Overlay all scenarios on spotlight map", value=st.session_state.presentation_view["show_all"])
        auto_dem = st.checkbox(
            "Allow auto-download DEM for missing cities",
            value=False,
            key="auto_dem_toggle_v2",
            help="Leave this off for stable switching. Turn it on only when preparing a city for the first time.",
        )
        horizon = st.slider("Forecast horizon (hours)", 1, 24, st.session_state.presentation_view["horizon"])
        hours_back = st.slider("History window (hours)", 48, 240, st.session_state.presentation_view["hours_back"], step=24)
        refresh = st.form_submit_button("Refresh Presentation View")

requested_view = {
    "api_base": api_base,
    "spotlight_city": spotlight_city,
    "compare_cities": compare_cities,
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

if not spotlight:
    st.error("No presentation data is currently loaded.")
    st.stop()

spotlight_scenario = next((s for s in spotlight.get("scenarios", []) if s["scenario"] == loaded_view["scenario"]), None)
spotlight_summary = selected_operational_summary(spotlight, loaded_view["scenario"])
spotlight_impact = selected_impact_summary(spotlight, loaded_view["scenario"])

detail_view = st.radio(
    "Presentation section",
    ["City Spotlight", "Operations", "Exposure", "Hotspots", "Forecast", "Compare"],
    horizontal=True,
    label_visibility="collapsed",
)

top_left, top_right = st.columns([1.25, 1.0], gap="large")

with top_left:
    if detail_view == "City Spotlight":
        st.markdown("### City Spotlight")
        st.markdown(
            f"{badge(spotlight.get('provider_label', 'n/a'), 'info')} "
            f"{badge(spotlight.get('support_tier', 'n/a'), support_tone(spotlight.get('support_tier', 'n/a')))} "
            f"{badge(spotlight.get('model', {}).get('forecast_mode_used', 'n/a'), 'neutral')}",
            unsafe_allow_html=True,
        )
        st.subheader(spotlight.get("display_name") or spotlight.get("city"))
        metric_cols = st.columns(4)
        metric_cols[0].metric("Peak", f"{float(spotlight.get('peak_prediction_m', 0.0)):.3f} m")
        metric_cols[1].metric(
            "Flooded",
            "n/a" if spotlight_scenario is None else f"{float(spotlight_scenario.get('flood_ratio', 0.0))*100:.2f}%",
        )
        metric_cols[2].metric("Risk", "n/a" if spotlight_scenario is None else str(spotlight_scenario.get("risk_level", "n/a")).upper())
        metric_cols[3].metric(
            "Delay",
            "n/a" if spotlight.get("source", {}).get("observation_delay_hours") is None else f"{float(spotlight['source']['observation_delay_hours']):.1f} h",
        )
        st.caption(f"Station {spotlight.get('station', 'n/a')} | Scenario {loaded_view['scenario']} | Horizon {loaded_view['horizon']}h")

    elif detail_view == "Operations":
        st.markdown("### Operations")
        if spotlight_summary:
            st.markdown(
                f"{badge(str(spotlight_summary.get('alert_level', 'n/a')).upper(), alert_tone(spotlight_summary.get('alert_level', 'n/a')))} "
                f"{badge(str(spotlight_summary.get('confidence', 'n/a')).upper(), 'neutral')}",
                unsafe_allow_html=True,
            )
        else:
            st.info("No operational summary is available for this scenario.")

        if spotlight_impact:
            st.markdown("#### Impact Snapshot")
            impact_cols = st.columns(4)
            impact_cols[0].metric(
                "Roads",
                "n/a" if spotlight_impact.get("affected_road_length_m") is None else f"{float(spotlight_impact['affected_road_length_m'])/1000.0:.2f} km",
            )
            impact_cols[1].metric("Priority Sites", f"{int(spotlight_impact.get('affected_site_count_total', 0))}")
            impact_cols[2].metric(
                "Population",
                "n/a" if spotlight_impact.get("population_affected_estimate") in {None, 0} else f"{int(spotlight_impact['population_affected_estimate']):,}",
            )
            impact_cols[3].metric("Categories", f"{int(spotlight_impact.get('categories_impacted', 0))}")

    elif detail_view == "Exposure":
        st.markdown("### Impacted Exposure")
        if spotlight_impact:
            render_impacted_site_types(spotlight_impact)
            rollup_rows = spotlight_impact.get("exposure_rollup", [])
            if rollup_rows:
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
                with st.expander("Exposure detail"):
                    exposure_df = pd.DataFrame(exposure_rows).copy()
                    for col in ["affected_area_m2", "affected_length_m"]:
                        if col in exposure_df.columns:
                            exposure_df[col] = exposure_df[col].map(lambda v: round(float(v), 1))
                    if "affected_value" in exposure_df.columns:
                        exposure_df["affected_value"] = exposure_df["affected_value"].map(lambda v: round(float(v), 1) if isinstance(v, float) else v)
                    st.dataframe(exposure_df, use_container_width=True, hide_index=True)
        else:
            st.info("No exposure summary is available for this scenario.")

    elif detail_view == "Hotspots":
        st.markdown("### Priority Hotspots")
        if spotlight_impact:
            hotspot_cols = st.columns(3)
            hotspot_cols[0].metric("Ranked", f"{int(spotlight_impact.get('hotspot_count', 0))}")
            hotspot_cols[1].metric(
                "Largest",
                "n/a" if spotlight_impact.get("largest_hotspot_area_m2") is None else f"{float(spotlight_impact['largest_hotspot_area_m2']):,.0f} m2",
            )
            hotspot_cols[2].metric("Layers", f"{int(spotlight_impact.get('exposure_layers_available', 0))}")
            hotspot_rows = spotlight_impact.get("top_hotspots", [])[:5]
            if hotspot_rows:
                hotspot_df = pd.DataFrame(
                    [
                        {
                            "rank": int(row["rank"]),
                            "risk": str(row["risk_level"]).upper(),
                            "score": round(float(row["priority_score"]), 1),
                            "area_m2": round(float(row["area_m2"]), 1),
                        }
                        for row in hotspot_rows
                    ]
                )
                st.dataframe(hotspot_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hotspot data is available for this scenario.")

    elif detail_view == "Forecast":
        st.markdown("### Forecast")
        forecast_points = spotlight.get("forecast", [])
        if forecast_points:
            forecast_df = pd.DataFrame(forecast_points)
            if {"p10_m", "p50_m", "p90_m"}.issubset(forecast_df.columns):
                st.line_chart(forecast_df, x="hour_ahead", y=["p10_m", "p50_m", "p90_m"])
            else:
                st.line_chart(forecast_df, x="hour_ahead", y="sea_level_m")
        else:
            st.info("No forecast trajectory is available.")
        if spotlight_scenario:
            footprint_cols = st.columns(3)
            footprint_cols[0].metric("Flood Area", f"{float(spotlight_scenario.get('flood_area_m2', 0.0)):,.0f} m2")
            footprint_cols[1].metric("Components", f"{int(spotlight_scenario.get('component_count', 0))}")
            footprint_cols[2].metric("Risk", str(spotlight_scenario.get("risk_level", "n/a")).upper())

    elif detail_view == "Compare":
        st.markdown("### Cross-city Compare")
        with st.spinner("Loading compare cities..."):
            compare_payloads, compare_warnings = load_compare_data(
                api_base=loaded_view["api_base"],
                compare_selection=loaded_view["compare_cities"],
                horizon=loaded_view["horizon"],
                hours_back=loaded_view["hours_back"],
                auto_dem=False,
                selected_scenario=loaded_view["scenario"],
            )
        compare_rows = [scenario_summary_row(payload, loaded_view["scenario"]) for payload in compare_payloads]
        if compare_rows:
            compare_df = pd.DataFrame(compare_rows).sort_values(
                ["support_tier", "flood_ratio_pct", "peak_prediction_m"],
                ascending=[True, False, False],
                na_position="last",
            )
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
            risk_df = compare_df[["city", "flood_ratio_pct", "flood_area_m2"]].set_index("city")
            if not risk_df.empty:
                st.bar_chart(risk_df)
        else:
            st.info("Select compare cities in the sidebar, then refresh the view.")

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

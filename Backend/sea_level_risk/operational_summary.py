from __future__ import annotations

from datetime import datetime


LEVEL_RANK = {"low": 0, "moderate": 1, "high": 2, "critical": 3}

DEFAULT_ALERT_THRESHOLDS_M = {
    "moderate": 0.30,
    "high": 0.60,
    "critical": 0.90,
}


def normalize_alert_thresholds(city_cfg: dict | None) -> dict:
    cfg = city_cfg or {}
    raw = cfg.get("alert_thresholds_m") or {}
    return {
        "moderate": float(raw.get("moderate", DEFAULT_ALERT_THRESHOLDS_M["moderate"])),
        "high": float(raw.get("high", DEFAULT_ALERT_THRESHOLDS_M["high"])),
        "critical": float(raw.get("critical", DEFAULT_ALERT_THRESHOLDS_M["critical"])),
    }


def pick_operational_scenario(payload: dict, preferred: str | None = None) -> str | None:
    scenarios = payload.get("scenarios", [])
    if not scenarios:
        return None

    names = [item.get("scenario") for item in scenarios if item.get("scenario")]
    if preferred in names:
        return preferred
    if "plus_50cm" in names:
        return "plus_50cm"
    if "plus_100cm" in names:
        return "plus_100cm"
    return names[0]


def _stage_level(peak_prediction_m: float, thresholds: dict) -> str:
    if peak_prediction_m >= thresholds["critical"]:
        return "critical"
    if peak_prediction_m >= thresholds["high"]:
        return "high"
    if peak_prediction_m >= thresholds["moderate"]:
        return "moderate"
    return "low"


def _confidence_level(payload: dict) -> str:
    support_tier = payload.get("support_tier")
    delay_h = payload.get("source", {}).get("observation_delay_hours")
    delay_h = float(delay_h) if delay_h is not None else None

    if support_tier == "official_realtime":
        if delay_h is None or delay_h <= 2:
            return "high"
        if delay_h <= 6:
            return "medium"
        return "low"

    if support_tier == "experimental_realtime":
        if delay_h is None or delay_h <= 2:
            return "medium"
        return "low"

    return "low"


def _headline(city_name: str, level: str, horizon_hours: int, scenario_name: str | None) -> str:
    prefix = {
        "low": "Low coastal flood risk",
        "moderate": "Elevated coastal flood risk",
        "high": "High coastal flood risk",
        "critical": "Critical coastal flood risk",
    }[level]
    if scenario_name:
        return f"{prefix} for {city_name} over the next {horizon_hours} hours under the {scenario_name} scenario."
    return f"{prefix} for {city_name} over the next {horizon_hours} hours."


def _recommended_actions(level: str, payload: dict, scenario: dict | None) -> list[str]:
    actions_by_level = {
        "low": [
            "Continue routine gauge and forecast monitoring.",
            "No immediate escalation is indicated from the current prototype thresholds.",
        ],
        "moderate": [
            "Increase monitoring frequency for low-lying coastal assets and nuisance-flooding locations.",
            "Review the next forecast cycle before the expected peak window.",
        ],
        "high": [
            "Notify operations staff for targeted field checks at low-lying coastal hotspots.",
            "Prepare partner or duty-officer briefings for possible localized coastal impacts.",
        ],
        "critical": [
            "Escalate to duty leadership for warning review and protective action planning.",
            "Prioritize hotspot inspection, interagency coordination, and public-facing briefing preparation.",
        ],
    }

    actions = list(actions_by_level[level])
    support_tier = payload.get("support_tier")
    if support_tier == "proxy_delayed":
        actions.append("Treat this city as proxy guidance only and confirm conditions with local sources before action.")
    elif support_tier == "experimental_realtime":
        actions.append("Use the feed as experimental guidance and verify high-impact decisions against additional observations.")

    if scenario and int(scenario.get("component_count", 0)) > 100:
        actions.append("Review coastal component clusters and ranked hotspots to prioritize operational attention.")

    return actions


def build_operational_summary(payload: dict, city_cfg: dict | None = None, scenario_name: str | None = None) -> dict:
    thresholds = normalize_alert_thresholds(city_cfg)
    scenario_name = pick_operational_scenario(payload, preferred=scenario_name)
    scenario = next((item for item in payload.get("scenarios", []) if item.get("scenario") == scenario_name), None)

    peak_prediction_m = float(payload.get("peak_prediction_m", 0.0))
    stage_level = _stage_level(peak_prediction_m, thresholds)
    flood_level = (scenario or {}).get("risk_level", "low")
    final_level = max(stage_level, flood_level, key=lambda item: LEVEL_RANK.get(item, 0))

    display_name = payload.get("display_name") or payload.get("city") or "Selected city"
    horizon_hours = int(payload.get("horizon_hours", 0))
    delay_h = payload.get("source", {}).get("observation_delay_hours")
    delay_h = None if delay_h is None else float(delay_h)
    confidence = _confidence_level(payload)

    flood_ratio_pct = None
    flood_area_m2 = None
    component_count = None
    if scenario:
        flood_ratio_pct = float(scenario.get("flood_ratio", 0.0)) * 100.0
        flood_area_m2 = float(scenario.get("flood_area_m2", 0.0))
        component_count = int(scenario.get("component_count", 0))

    summary_text = (
        f"Peak forecast reaches {peak_prediction_m:.3f} m."
        + (
            f" Under {scenario_name}, mapped flooding covers {flood_ratio_pct:.2f}% of land "
            f"({flood_area_m2:,.0f} m2) across {component_count} coastal components."
            if scenario is not None
            else " No flood scenario output is available for the current city selection."
        )
    )

    return {
        "scenario_basis": scenario_name,
        "alert_level": final_level,
        "stage_level": stage_level,
        "scenario_risk_level": flood_level,
        "confidence": confidence,
        "confidence_reason": (
            "Official near-real-time feed with low observation delay."
            if confidence == "high"
            else "Use as operational context, but verify against additional observations before escalation."
        ),
        "headline": _headline(display_name, final_level, horizon_hours, scenario_name),
        "summary_text": summary_text,
        "recommended_actions": _recommended_actions(final_level, payload, scenario),
        "peak_prediction_m": peak_prediction_m,
        "observation_delay_hours": delay_h,
        "flood_ratio_pct": flood_ratio_pct,
        "flood_area_m2": flood_area_m2,
        "component_count": component_count,
        "thresholds_m": thresholds,
        "thresholds_note": (city_cfg or {}).get(
            "alert_thresholds_note",
            "Prototype operational thresholds for internal monitoring; not an official warning standard.",
        ),
    }


def build_briefing_markdown(payload: dict, city_cfg: dict | None = None, scenario_name: str | None = None) -> str:
    summary = build_operational_summary(payload, city_cfg=city_cfg, scenario_name=scenario_name)
    generated_at = payload.get("generated_at_utc") or datetime.utcnow().isoformat()
    provider = payload.get("provider_label", "n/a")
    station = payload.get("station", "n/a")
    display_name = payload.get("display_name") or payload.get("city") or "Selected city"
    observation_delay = "n/a"
    if summary["observation_delay_hours"] is not None:
        observation_delay = f"{summary['observation_delay_hours']:.1f} h"
    flood_ratio = "n/a"
    if summary["flood_ratio_pct"] is not None:
        flood_ratio = f"{summary['flood_ratio_pct']:.2f}%"
    flood_area = "n/a"
    if summary["flood_area_m2"] is not None:
        flood_area = f"{summary['flood_area_m2']:,.0f} m2"
    component_count = "n/a" if summary["component_count"] is None else str(summary["component_count"])

    lines = [
        f"# Coastal Flood Risk Briefing: {display_name}",
        "",
        f"- Generated (UTC): `{generated_at}`",
        f"- Provider: `{provider}`",
        f"- Station: `{station}`",
        f"- Forecast mode: `{payload.get('model', {}).get('forecast_mode_used', 'n/a')}`",
        f"- Scenario basis: `{summary.get('scenario_basis') or 'none'}`",
        f"- Alert level: `{str(summary.get('alert_level', 'unknown')).upper()}`",
        f"- Confidence: `{str(summary.get('confidence', 'unknown')).upper()}`",
        "",
        "## Headline",
        summary["headline"],
        "",
        "## Situation Summary",
        summary["summary_text"],
        "",
        "## Key Metrics",
        f"- Peak forecast: `{summary['peak_prediction_m']:.3f} m`",
        f"- Observation delay: `{observation_delay}`",
        f"- Flood ratio: `{flood_ratio}`",
        f"- Flood area: `{flood_area}`",
        f"- Coastal components: `{component_count}`",
        "",
        "## Recommended Actions",
    ]
    lines.extend(f"- {item}" for item in summary["recommended_actions"])
    lines.extend(
        [
            "",
            "## Threshold Note",
            summary["thresholds_note"],
            "",
            "## Source Notes",
            payload.get("city_notes") or "No additional city note provided.",
        ]
    )
    source_note = payload.get("source", {}).get("note")
    if source_note:
        lines.append(f"- Source status note: {source_note}")

    return "\n".join(lines)

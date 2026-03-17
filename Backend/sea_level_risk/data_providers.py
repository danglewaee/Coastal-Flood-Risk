from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape

import pandas as pd
import requests


NOAA_ENDPOINT = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
IOC_BGRAPH_ENDPOINT = "https://www.ioc-sealevelmonitoring.org/bgraph.php"
IOC_PERIODS_DAYS = [0.5, 1, 7, 30]
IOC_VALUE_PRIORITY = ["wls", "flt", "prs", "rad", "pr2"]


@dataclass
class SeriesFetchResult:
    frame: pd.DataFrame
    provider: str
    station_ref: str
    source_name: str
    observation_delay_hours: float | None
    source_value_column: str | None = None
    status: str = "ok"
    note: str | None = None


def _normalize_hourly_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "sea_level"])

    hourly = (
        df.set_index("timestamp")["sea_level"]
        .sort_index()
        .resample("1h")
        .mean()
        .interpolate(limit=6, limit_direction="both")
        .dropna()
        .reset_index()
    )
    return hourly


def _raise_for_json_error(response: requests.Response) -> dict:
    payload = response.json()
    if isinstance(payload, dict) and "error" in payload:
        error_obj = payload["error"]
        if isinstance(error_obj, dict):
            message = error_obj.get("message", error_obj)
        else:
            message = error_obj
        raise RuntimeError(f"Provider error: {message}")
    return payload


def fetch_noaa_series(
    station: str,
    begin_date: str,
    end_date: str,
    product: str = "water_level",
    datum: str = "MSL",
    units: str = "metric",
    time_zone: str = "gmt",
    interval: str = "h",
) -> pd.DataFrame:
    params = {
        "product": product,
        "application": "sea_level_risk_pipeline",
        "begin_date": begin_date,
        "end_date": end_date,
        "datum": datum,
        "station": station,
        "time_zone": time_zone,
        "units": units,
        "format": "json",
    }
    if product in {"water_level", "predictions"} and interval:
        params["interval"] = interval

    response = requests.get(NOAA_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    payload = _raise_for_json_error(response)

    rows = payload.get("data") or payload.get("predictions") or []
    if not rows:
        return pd.DataFrame(columns=["timestamp", "sea_level"])

    df = pd.DataFrame(rows)
    df = df.rename(columns={"t": "timestamp", "v": "sea_level"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["sea_level"] = pd.to_numeric(df["sea_level"], errors="coerce")
    return df[["timestamp", "sea_level"]].dropna().sort_values("timestamp")


def fetch_noaa_recent(
    station: str,
    hours_back: int,
    datum: str = "MSL",
    product: str = "water_level",
    now_utc: datetime | None = None,
) -> SeriesFetchResult:
    end_ts = now_utc or datetime.now(timezone.utc)
    begin_ts = end_ts - timedelta(hours=max(hours_back, 24) + 24)

    chunks = []
    chunk_start = begin_ts
    while chunk_start <= end_ts:
        chunk_end = min(end_ts, chunk_start + timedelta(days=2))
        chunk = fetch_noaa_series(
            station=station,
            begin_date=chunk_start.strftime("%Y%m%d"),
            end_date=chunk_end.strftime("%Y%m%d"),
            product=product,
            datum=datum,
            interval="h",
        )
        chunks.append(chunk)
        chunk_start = chunk_end + timedelta(days=1)

    current = (
        pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        if chunks
        else pd.DataFrame(columns=["timestamp", "sea_level"])
    )
    current = current[current["timestamp"] >= pd.Timestamp(begin_ts)]
    if not current.empty:
        current = _normalize_hourly_frame(current)
        last_obs = pd.Timestamp(current["timestamp"].iloc[-1])
        delay_hours = max(0.0, (end_ts - last_obs.to_pydatetime()).total_seconds() / 3600.0)
        return SeriesFetchResult(
            frame=current.tail(max(hours_back, 24)).reset_index(drop=True),
            provider="noaa",
            station_ref=station,
            source_name="NOAA CO-OPS",
            observation_delay_hours=delay_hours,
            source_value_column="sea_level",
            status="ok",
        )

    fallback_end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    fallback_begin = fallback_end - timedelta(hours=max(hours_back, 24) + 24)
    fallback = fetch_noaa_series(
        station=station,
        begin_date=fallback_begin.strftime("%Y%m%d"),
        end_date=fallback_end.strftime("%Y%m%d"),
        product=product,
        datum=datum,
        interval="h",
    )
    fallback = fallback[fallback["timestamp"] >= pd.Timestamp(fallback_begin)]
    fallback = _normalize_hourly_frame(fallback)
    if fallback.empty:
        raise RuntimeError(f"No NOAA data returned for station {station}.")

    return SeriesFetchResult(
        frame=fallback.tail(max(hours_back, 24)).reset_index(drop=True),
        provider="noaa",
        station_ref=station,
        source_name="NOAA CO-OPS",
        observation_delay_hours=None,
        source_value_column="sea_level",
        status="fallback_archived",
        note="Realtime NOAA window was empty, using archived fallback window ending 2024-12-31 UTC.",
    )


def _sanitize_ioc_column_name(name: str) -> str:
    clean = re.sub(r"\([^)]*\)", "", str(name))
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", clean).strip("_").lower()
    return clean


def _parse_ioc_table(html_text: str, preferred_value_column: str | None = None) -> tuple[pd.DataFrame, str | None]:
    table_match = re.search(r"<table[^>]*>(.*?)</table>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if not table_match:
        return pd.DataFrame(columns=["timestamp", "sea_level"]), None

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_match.group(1), flags=re.IGNORECASE | re.DOTALL)
    parsed_rows: list[list[str]] = []
    for row_html in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
        if cells:
            cleaned = [unescape(re.sub(r"<[^>]+>", "", cell)).strip() for cell in cells]
            parsed_rows.append(cleaned)

    if len(parsed_rows) < 2:
        return pd.DataFrame(columns=["timestamp", "sea_level"]), None

    header = parsed_rows[0]
    data_rows = [row for row in parsed_rows[1:] if len(row) == len(header)]
    if not data_rows:
        return pd.DataFrame(columns=["timestamp", "sea_level"]), None

    raw = pd.DataFrame(data_rows, columns=[_sanitize_ioc_column_name(col) for col in header])
    time_col = next((c for c in raw.columns if "time" in c), None)

    candidate_order = []
    if preferred_value_column:
        candidate_order.append(preferred_value_column.lower())
    candidate_order.extend([c for c in IOC_VALUE_PRIORITY if c not in candidate_order])

    value_col = None
    for candidate in candidate_order:
        if candidate in raw.columns:
            value_col = candidate
            break

    if not value_col:
        numeric_columns = [c for c in raw.columns if c != time_col]
        value_col = numeric_columns[0] if numeric_columns else None

    if not value_col:
        return pd.DataFrame(columns=["timestamp", "sea_level"]), None

    df = raw[[time_col, value_col]].copy()
    df = df.rename(columns={time_col: "timestamp", value_col: "sea_level"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["sea_level"] = pd.to_numeric(df["sea_level"], errors="coerce")
    df = df.dropna().sort_values("timestamp")
    return df[["timestamp", "sea_level"]], value_col


def fetch_ioc_recent(
    station_code: str,
    hours_back: int,
    preferred_value_column: str | None = None,
    now_utc: datetime | None = None,
) -> SeriesFetchResult:
    end_ts = now_utc or datetime.now(timezone.utc)
    latest_valid: pd.DataFrame | None = None
    latest_column = preferred_value_column
    latest_period = None

    for period in IOC_PERIODS_DAYS:
        response = requests.get(
            IOC_BGRAPH_ENDPOINT,
            params={"code": station_code, "output": "tab", "period": period},
            timeout=60,
        )
        response.raise_for_status()
        text = response.text
        if "NO DATA" in text and "<table" not in text:
            continue

        parsed, source_value_column = _parse_ioc_table(text, preferred_value_column=preferred_value_column)
        if parsed.empty:
            continue

        latest_valid = parsed
        latest_column = source_value_column
        latest_period = period

        last_obs = pd.Timestamp(parsed["timestamp"].iloc[-1]).to_pydatetime()
        delay_hours = max(0.0, (end_ts - last_obs).total_seconds() / 3600.0)
        if delay_hours <= max(6.0, hours_back / 3):
            break

    if latest_valid is None or latest_valid.empty:
        raise RuntimeError(f"No IOC data returned for station code '{station_code}'.")

    hourly = _normalize_hourly_frame(latest_valid)
    if hourly.empty:
        raise RuntimeError(f"IOC station '{station_code}' returned no usable numeric water-level values.")

    last_obs = pd.Timestamp(hourly["timestamp"].iloc[-1]).to_pydatetime()
    delay_hours = max(0.0, (end_ts - last_obs).total_seconds() / 3600.0)
    status = "ok" if delay_hours <= 6.0 else "delayed"
    note = None
    if latest_period == 30 and delay_hours > 24.0:
        note = "Latest IOC rows were only available in the 30-day window. Treat this city as delayed/proxy, not true realtime."

    return SeriesFetchResult(
        frame=hourly.tail(max(hours_back, 24)).reset_index(drop=True),
        provider="ioc",
        station_ref=station_code,
        source_name="UNESCO IOC Sea Level Monitoring",
        observation_delay_hours=delay_hours,
        source_value_column=latest_column,
        status=status,
        note=note,
    )


def ensure_hours_back_coverage(df: pd.DataFrame, hours_back: int) -> pd.DataFrame:
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        raise ValueError("Expected a dataframe with a 'timestamp' column.")

    cutoff = pd.Timestamp(df["timestamp"].max()) - pd.Timedelta(hours=hours_back)
    trimmed = df[df["timestamp"] >= cutoff].copy()
    return trimmed.reset_index(drop=True) if not trimmed.empty else df.tail(max(hours_back, 24)).reset_index(drop=True)

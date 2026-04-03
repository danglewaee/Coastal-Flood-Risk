from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from .city_registry import load_city_registry


ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_HOURLY_VARS = {
    "wind_speed_10m": "wind_speed",
    "pressure_msl": "air_pressure",
    "precipitation": "precipitation",
}


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _infer_date_range_from_csv(csv_path: str, *, time_col: str = "timestamp") -> tuple[date, date]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Hourly water-level CSV does not exist: {path}")
    df = pd.read_csv(path, usecols=[time_col])
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True).dropna().sort_values()
    if ts.empty:
        raise ValueError(f"Could not infer driver date range from '{csv_path}': no valid timestamps in '{time_col}'.")
    return ts.iloc[0].date(), ts.iloc[-1].date()


def _iter_chunks(start_date: date, end_date: date, *, max_days: int = 366) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=max_days - 1), end_date)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


def _request_open_meteo_chunk(
    *,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    hourly_vars: dict[str, str],
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(hourly_vars.keys()),
        "timezone": "UTC",
    }
    response = requests.get(ARCHIVE_API_URL, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly") or {}
    timestamps = hourly.get("time")
    if not timestamps:
        raise ValueError(
            f"Open-Meteo archive response did not contain hourly timestamps for range "
            f"{start_date.isoformat()} to {end_date.isoformat()}."
        )

    frame = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, errors="coerce", utc=True)})
    for source_name, target_name in hourly_vars.items():
        values = hourly.get(source_name)
        if values is None:
            raise ValueError(f"Open-Meteo archive response is missing hourly variable '{source_name}'.")
        frame[target_name] = pd.to_numeric(pd.Series(values), errors="coerce")

    return frame.dropna(subset=["timestamp"]).reset_index(drop=True)


def download_weather_drivers(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    out_csv: str,
    city_key: str | None = None,
    timezone_name: str = "UTC",
    source_label: str = "Open-Meteo Historical Weather API",
) -> dict:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date.")

    frames = [
        _request_open_meteo_chunk(
            latitude=latitude,
            longitude=longitude,
            start_date=chunk_start,
            end_date=chunk_end,
            hourly_vars=DEFAULT_HOURLY_VARS,
        )
        for chunk_start, chunk_end in _iter_chunks(start, end)
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    combined["river_discharge"] = 0.0
    combined = combined[["timestamp", "wind_speed", "air_pressure", "precipitation", "river_discharge"]]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    metadata = {
        "city": city_key,
        "source": source_label,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone_name,
        "requested_start_date": start.isoformat(),
        "requested_end_date": end.isoformat(),
        "hourly_variables": list(DEFAULT_HOURLY_VARS.keys()),
        "output_columns": list(combined.columns),
        "rows": int(combined.shape[0]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "river_discharge is currently zero-filled because a generic city-scale discharge archive is not wired in yet.",
            "Timestamps are normalized to UTC to align with hourly gauge histories.",
        ],
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"csv": str(out_path), "metadata_json": str(meta_path), "rows": int(combined.shape[0])}


def main():
    parser = argparse.ArgumentParser(description="Download hourly weather drivers for multivariate_v2 city training.")
    parser.add_argument("--city", default=None, help="City key from city_registry.json.")
    parser.add_argument("--lat", type=float, default=None, help="Latitude override if not using --city.")
    parser.add_argument("--lon", type=float, default=None, help="Longitude override if not using --city.")
    parser.add_argument("--start-date", default=None, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--match-csv",
        default=None,
        help="Optional hourly sea-level CSV used to infer start/end dates when not passed explicitly.",
    )
    parser.add_argument("--match-time-col", default="timestamp")
    parser.add_argument("--out", default=None, help="Output CSV path. Defaults to data/<city>_drivers_hourly.csv.")
    args = parser.parse_args()

    city_key = args.city.strip().lower() if args.city else None
    timezone_name = "UTC"
    latitude = args.lat
    longitude = args.lon

    if city_key:
        registry = load_city_registry()
        if city_key not in registry:
            raise ValueError(f"Unknown city '{city_key}'.")
        cfg = registry[city_key]
        latitude = cfg.get("lat") if latitude is None else latitude
        longitude = cfg.get("lon") if longitude is None else longitude
        timezone_name = cfg.get("timezone") or timezone_name

    if latitude is None or longitude is None:
        raise ValueError("Provide either --city or both --lat and --lon.")

    inferred_start = inferred_end = None
    if args.match_csv:
        inferred_start, inferred_end = _infer_date_range_from_csv(args.match_csv, time_col=args.match_time_col)

    start_date = args.start_date or (inferred_start.isoformat() if inferred_start else None)
    end_date = args.end_date or (inferred_end.isoformat() if inferred_end else None)
    if not start_date or not end_date:
        raise ValueError("Provide --start-date/--end-date or --match-csv to infer them.")

    if args.out:
        out_csv = args.out
    elif city_key:
        out_csv = f"data/{city_key}_drivers_hourly.csv"
    else:
        out_csv = "data/weather_drivers_hourly.csv"

    result = download_weather_drivers(
        latitude=float(latitude),
        longitude=float(longitude),
        start_date=start_date,
        end_date=end_date,
        out_csv=out_csv,
        city_key=city_key,
        timezone_name=timezone_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

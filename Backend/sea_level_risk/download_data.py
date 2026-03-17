import argparse
from pathlib import Path

import pandas as pd

from .data_providers import fetch_noaa_series


def _parse_ymd(date_str: str) -> pd.Timestamp:
    ts = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date '{date_str}'. Expected YYYYMMDD.")
    return ts


def _fetch_noaa_chunk(
    station: str,
    begin_date: str,
    end_date: str,
    product: str,
    datum: str,
    units: str,
    time_zone: str,
) -> pd.DataFrame:
    return fetch_noaa_series(
        station=station,
        begin_date=begin_date,
        end_date=end_date,
        product=product,
        datum=datum,
        units=units,
        time_zone=time_zone,
        interval="h",
    )


def download_noaa_hourly(
    station: str,
    begin_date: str,
    end_date: str,
    out_csv: str,
    product: str = "water_level",
    datum: str = "MSL",
    units: str = "metric",
    time_zone: str = "gmt",
) -> str:
    start_ts = _parse_ymd(begin_date)
    end_ts = _parse_ymd(end_date)
    if end_ts < start_ts:
        raise ValueError("end_date must be >= begin_date")

    year_starts = pd.date_range(start=start_ts, end=end_ts, freq="YS")
    if len(year_starts) == 0:
        year_starts = pd.DatetimeIndex([pd.Timestamp(year=start_ts.year, month=1, day=1)])

    chunks = []
    for ys in year_starts:
        chunk_start = max(start_ts, ys)
        chunk_end = min(end_ts, pd.Timestamp(year=ys.year, month=12, day=31))
        if chunk_start > chunk_end:
            continue

        chunk_begin = chunk_start.strftime("%Y%m%d")
        chunk_finish = chunk_end.strftime("%Y%m%d")

        df_chunk = _fetch_noaa_chunk(
            station=station,
            begin_date=chunk_begin,
            end_date=chunk_finish,
            product=product,
            datum=datum,
            units=units,
            time_zone=time_zone,
        )
        chunks.append(df_chunk)

    if not chunks:
        raise RuntimeError("No NOAA data returned for requested period.")

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)


def normalize_uhslc_file(input_path: str, out_csv: str, time_col: str = "timestamp", value_col: str = "sea_level") -> str:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(str(src))

    if src.suffix.lower() in {".csv", ".txt", ".dat"}:
        df = pd.read_csv(src)
    else:
        raise ValueError("Unsupported UHSLC file format. Use csv/txt/dat.")

    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Input must contain columns '{time_col}' and '{value_col}'.")

    out = df[[time_col, value_col]].copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna().sort_values(time_col).rename(columns={time_col: "timestamp", value_col: "sea_level"})

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Download/normalize sea-level time series data")
    subparsers = parser.add_subparsers(dest="source", required=True)

    noaa_parser = subparsers.add_parser("noaa")
    noaa_parser.add_argument("--station", required=True, help="NOAA station id, e.g. 1612340")
    noaa_parser.add_argument("--begin", required=True, help="YYYYMMDD")
    noaa_parser.add_argument("--end", required=True, help="YYYYMMDD")
    noaa_parser.add_argument("--out", required=True)
    noaa_parser.add_argument("--datum", default="MSL")
    noaa_parser.add_argument("--product", default="water_level")

    uhslc_parser = subparsers.add_parser("uhslc")
    uhslc_parser.add_argument("--input", required=True)
    uhslc_parser.add_argument("--out", required=True)
    uhslc_parser.add_argument("--time-col", default="timestamp")
    uhslc_parser.add_argument("--value-col", default="sea_level")

    args = parser.parse_args()

    if args.source == "noaa":
        out_path = download_noaa_hourly(
            args.station,
            args.begin,
            args.end,
            args.out,
            product=args.product,
            datum=args.datum,
        )
    else:
        out_path = normalize_uhslc_file(args.input, args.out, args.time_col, args.value_col)

    print({"output_csv": out_path})


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import rasterio


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def check_alignment(dem_path: Path, depth_path: Path) -> int:
    if not dem_path.exists():
        print(f"DEM missing: {dem_path}")
        return 2
    if not depth_path.exists():
        print(f"Depth raster missing: {depth_path}")
        return 2

    with rasterio.open(dem_path) as dem_src, rasterio.open(depth_path) as depth_src:
        mismatches = []
        if dem_src.crs != depth_src.crs:
            mismatches.append(("crs", dem_src.crs, depth_src.crs))
        if dem_src.transform != depth_src.transform:
            mismatches.append(("transform", dem_src.transform, depth_src.transform))
        if dem_src.width != depth_src.width or dem_src.height != depth_src.height:
            mismatches.append(("shape", (dem_src.width, dem_src.height), (depth_src.width, depth_src.height)))

        print("DEM:", dem_path)
        print("DEPTH:", depth_path)
        print(f"DEM CRS: {_format_value(dem_src.crs)}")
        print(f"DEPTH CRS: {_format_value(depth_src.crs)}")
        print(f"DEM transform: {_format_value(dem_src.transform)}")
        print(f"DEPTH transform: {_format_value(depth_src.transform)}")
        print(f"DEM shape: {dem_src.width}x{dem_src.height}")
        print(f"DEPTH shape: {depth_src.width}x{depth_src.height}")

        if not mismatches:
            print("Alignment: OK")
            return 0

        print("Alignment: MISMATCH")
        for name, a, b in mismatches:
            print(f"- {name}: DEM={a} | DEPTH={b}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Check DEM vs depth raster alignment.")
    parser.add_argument("--dem", required=True, help="Path to DEM GeoTIFF")
    parser.add_argument("--depth", required=True, help="Path to depth raster GeoTIFF")
    args = parser.parse_args()

    exit_code = check_alignment(Path(args.dem), Path(args.depth))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

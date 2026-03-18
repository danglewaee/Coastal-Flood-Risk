from __future__ import annotations

import argparse
from pathlib import Path

from .hydro_postprocess import main as postprocess_main


def _run_postprocess(args_list: list[str]) -> None:
    import sys

    argv_backup = sys.argv[:]
    try:
        sys.argv = args_list
        postprocess_main()
    finally:
        sys.argv = argv_backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch hydro postprocess for plus_20/50/100cm.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--dem", required=True)
    parser.add_argument("--depth-threshold", type=float, default=0.01)
    parser.add_argument("--min-land-elevation", type=float, default=0.0)
    parser.add_argument("--min-component-area", type=float, default=2500.0)
    parser.add_argument("--smooth", type=float, default=10.0)
    args = parser.parse_args()

    depth_dir = Path(args.depth_dir)
    scenarios = {
        "plus_20cm": depth_dir / "plus_20cm_depth.tif",
        "plus_50cm": depth_dir / "plus_50cm_depth.tif",
        "plus_100cm": depth_dir / "plus_100cm_depth.tif",
    }

    for scenario, depth_path in scenarios.items():
        if not depth_path.exists():
            print(f"Skip {scenario}: missing {depth_path}")
            continue
        _run_postprocess(
            [
                "hydro_postprocess",
                "--city",
                args.city,
                "--scenario",
                scenario,
                "--depth",
                str(depth_path),
                "--dem",
                str(args.dem),
                "--depth-threshold",
                str(args.depth_threshold),
                "--min-land-elevation",
                str(args.min_land_elevation),
                "--min-component-area",
                str(args.min_component_area),
                "--smooth",
                str(args.smooth),
            ]
        )


if __name__ == "__main__":
    main()

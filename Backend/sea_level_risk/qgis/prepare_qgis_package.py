import argparse
import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_package(city: str, dem_path: str, realtime_dir: str, out_root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / f"{city}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_src = Path(dem_path)
    rt_dir = Path(realtime_dir)

    copy_if_exists(dem_src, out_dir / "layers" / dem_src.name)

    for name in ["flood_plus_20cm.geojson", "flood_plus_50cm.geojson", "flood_plus_100cm.geojson", "hotspots.geojson"]:
        copy_if_exists(rt_dir / name, out_dir / "layers" / name)

    style_dir = Path("Backend/sea_level_risk/qgis/styles")
    for qml in style_dir.glob("*.qml"):
        copy_if_exists(qml, out_dir / "styles" / qml.name)

    gpkg_path = out_dir / "layers" / f"{city}_anti_flood.gpkg"
    for layer_file, layer_name in [
        ("flood_plus_20cm.geojson", "flood_plus_20cm"),
        ("flood_plus_50cm.geojson", "flood_plus_50cm"),
        ("flood_plus_100cm.geojson", "flood_plus_100cm"),
        ("hotspots.geojson", "hotspots"),
    ]:
        p = out_dir / "layers" / layer_file
        if p.exists():
            gdf = gpd.read_file(p)
            if not gdf.empty:
                gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")

    quickstart = out_dir / "QGIS_QUICKSTART.txt"
    repo_root = Path.cwd().resolve()
    console_snippet = out_dir / "CREATE_TEMPLATE_IN_QGIS_CONSOLE.py"
    console_snippet.write_text(
        "import sys\n"
        f"sys.path.append(r\"{repo_root}\")\n"
        "from Backend.sea_level_risk.qgis.create_qgis_template import create_project\n"
        f"pkg = r\"{out_dir.resolve()}\"\n"
        "out_qgz = create_project(pkg)\n"
        "print(out_qgz)\n",
        encoding="utf-8",
    )

    quickstart.write_text(
        "QGIS Quickstart\n"
        "1. Add raster: layers/<dem>.tif\n"
        "2. Add vector layers from layers/*.geojson or *.gpkg\n"
        "3. Apply styles in styles/*.qml (Layer Properties -> Symbology -> Style -> Load Style).\n"
        "4. Recommended order: dem -> flood_20 -> flood_50 -> flood_100 -> hotspots\n"
        "5. Flood layers are coast-connected DEM threshold outputs with small polygons already filtered.\n",
        encoding="utf-8",
    )
    quickstart.write_text(
        quickstart.read_text(encoding="utf-8")
        + "6. To generate anti_flood_template.qgz, open QGIS Python Console and run:\n"
        + f"   exec(open(r\"{console_snippet.resolve()}\", encoding=\"utf-8\").read())\n",
        encoding="utf-8",
    )

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare standardized QGIS package for anti-flood analysis")
    parser.add_argument("--city", default="honolulu")
    parser.add_argument("--dem", default="data/honolulu_dem.tif")
    parser.add_argument("--realtime-dir", default="Backend/sea_level_risk/outputs/realtime/honolulu")
    parser.add_argument("--out-root", default="Backend/sea_level_risk/outputs/qgis_packages")
    args = parser.parse_args()

    out = prepare_package(args.city, args.dem, args.realtime_dir, args.out_root)
    print({"package_dir": str(out)})


if __name__ == "__main__":
    main()

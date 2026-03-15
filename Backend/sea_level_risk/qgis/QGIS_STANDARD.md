# QGIS Standardization Guide

## 1) Build a QGIS package from cleaned scenario outputs
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.qgis.prepare_qgis_package \
  --city honolulu \
  --dem data/honolulu_dem.tif \
  --realtime-dir Backend/sea_level_risk/outputs \
  --out-root Backend/sea_level_risk/outputs/qgis_packages
```

This creates:
- `layers/` with DEM + flood/hotspots layers
- `styles/` with `.qml` styles
- `<city>_anti_flood.gpkg` (single-file vector package)
- `QGIS_QUICKSTART.txt`

## 2) Load in QGIS
1. Add raster DEM from `layers/`.
2. Add vectors from `*.gpkg` or `*.geojson`.
3. Apply styles from `styles/`:
- `flood_plus_20cm.qml`
- `flood_plus_50cm.qml`
- `flood_plus_100cm.qml`
- `hotspots.qml`

## 3) Recommended layer order
1. DEM
2. flood_plus_20cm
3. flood_plus_50cm
4. flood_plus_100cm
5. hotspots

## 4) CRS standard
- Display CRS: EPSG:4326 (or local projected CRS for planning maps).
- Area computations are done in equal-area projection in backend.

## 5) Layout export standard
- Title: `Anti-Flood Priority Map - <City> - <Date>`
- Include: legend, scale bar, north arrow, source credits.
- Export: PNG (300 dpi) and PDF.
- Preferred base layers: cleaned flood GeoJSON/GPKG created by the backend, not manual raster polygonization.

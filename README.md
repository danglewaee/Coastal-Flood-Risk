# Coastal Flood Risk

Interactive coastal flood exploration with React, Leaflet, Flask and the existing Python forecasting/GIS pipeline. No QGIS installation or map API key is required.

## Run

From the repository root, start the Explorer API in one terminal:

```powershell
Backend\.venv311\Scripts\python -m Backend.sea_level_risk.explorer_api --port 8100
```

Start the frontend in another terminal (Node 22.12+ or 24):

```powershell
cd Frontend/react-app
npm ci
npm start
```

Open **http://127.0.0.1:5173**. The first view reads saved GIS assets; it does not download DEMs or run a model. Use **Refresh observations** explicitly to fetch a forecast. Python 3.11 dependencies: `Backend/requirements-base.txt`; forecasts additionally require `Backend/requirements-ml.txt`.

## Explore

- City spotlight: select location and +20/+50/+100 cm scenario.
- Map: click a flood polygon to inspect its area, affected roads/facilities, DEM samples and spatial overlap across scenarios.
- Operations and Exposure: measured spatial intersections, layer availability, search and affected-only filtering.
- Hotspots: select an area-ranked region and fly to it on the map.
- Forecast: saved forecasts with uncertainty bands, existing backtest reports and observation provenance.
- Compare: footprint and infrastructure exposure across saved scenarios.
- Share: URL includes page, city, scenario and selected region. Export a full footprint or selected region as GeoJSON.

## Data Required

Large input data and generated GIS/model assets remain outside Git. An empty clone shows missing-data states until assets are prepared. Existing local outputs are reused:

```text
Backend/sea_level_risk/outputs/realtime/<city>/flood_plus_*.geojson
Backend/sea_level_risk/outputs/realtime/<city>/flood_plus_*.meta.json
data/exposure/<city>/*.geojson
data/dem_city_cache/<city>_dem_clip.tif
```

City/layer definitions live in `city_registry.json` and `exposure_registry.json`. See [data preparation and backend tools](Backend/sea_level_risk/README.md). The React app also works with `/api` routes registered on the original `realtime_api` server.

## What The Numbers Mean

Flood footprints are scenario outputs from the existing coastal-connected DEM threshold workflow, or imported hydrodynamic outputs where configured. They are not annual flood probabilities. Point facilities are counted inside the footprint; line lengths use a local metric CRS; population is allocated by the flooded share of polygon area. Missing layers return `null`, distinct from zero exposure.

Map files are labelled saved analyses. File timestamps are not NOAA observation timestamps. A live request may receive a historical provider fallback; the Forecast page reports that status. DEM/water-level vertical alignment remains unverified and is not presented as a validated depth estimate. Region ranking in the explorer is by area; it is not an infrastructure-weighted model.

## Verify And Deploy

```powershell
Backend\.venv311\Scripts\python -m unittest discover -s Backend/tests -v
cd Frontend/react-app
npm run build
# With API and Vite already running; Microsoft Edge installed:
npm test
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for serving the built React app through Flask/Waitress or hosting the frontend separately. The original Streamlit presentation remains available at `Backend/sea_level_risk/presentation_app.py`.

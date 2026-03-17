# Realtime Coastal Water-Level and Flood Risk Pipeline

## Setup (Python 3.11)
```powershell
powershell -ExecutionPolicy Bypass -File Backend/setup_py311_env.ps1
Backend/.venv311/Scripts/python -m pip install -r Backend/requirements-ml.txt
```

## Train model (one-time)
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.train \
  --csv data/honolulu_hourly.csv \
  --value-col sea_level \
  --time-col timestamp \
  --model-type axial_lstm
```

## Realtime API (multi-city, provider-aware)
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.realtime_api \
  --model Backend/sea_level_risk/outputs/sea_level_axial_lstm.keras \
  --metadata Backend/sea_level_risk/outputs/metadata.json \
  --host 127.0.0.1 --port 8000
```

Endpoints:
- `GET /health`
- `GET /realtime/cities`
- `GET /realtime/forecast?city=honolulu&horizon=6&hours_back=96&auto_dem=1`
- `GET /realtime/forecast?city=boston&horizon=6&hours_back=96&auto_dem=1`
- `GET /realtime/forecast?city=newyork&horizon=6&hours_back=96&auto_dem=1`
- `GET /realtime/forecast?city=jakarta&horizon=6&hours_back=96&auto_dem=1`
- `GET /realtime/forecast?city=amsterdam&horizon=6&hours_back=96&auto_dem=1`
- `GET /realtime/forecast?provider=noaa&station=1612340&horizon=6&hours_back=96&auto_dem=1`

Notes:
- `city_registry.json` now stores provider/station metadata, support tier, proxy mode, forecast mode, and DEM hints per city.
- `honolulu` uses the trained deep-learning model. The new large-city entries currently use a tide-aware baseline until city-specific models are trained.
- NOAA realtime fetch uses `water_level`, not `hourly_height`.
- IOC realtime fetch parses the public HTML table feed and resamples it to hourly series.
- `amsterdam` is implemented as `Amsterdam-region proxy (Hoek van Holland)`. Treat it as delayed regional proxy mode, not a direct Amsterdam city gauge.
- If a city has no local DEM, `auto_dem=1` will fetch Copernicus DEM by NOAA station metadata or by city `lat/lon` from the registry.

## Supported city modes
- `honolulu`: official NOAA realtime + local deep model
- `boston`: official NOAA realtime + tide-aware baseline
- `newyork`: official NOAA realtime + tide-aware baseline
- `jakarta`: experimental IOC realtime + tide-aware baseline
- `amsterdam`: delayed IOC coastal proxy + tide-aware baseline

## Dashboard (interactive)
Run the API first, then run dashboard:
```powershell
Backend/.venv311/Scripts/streamlit run Backend/sea_level_risk/dashboard_app.py
```

Dashboard features:
- choose a coastal city
- fetch provider-aware realtime data
- render a practical 2D flood polygon map on a real basemap
- render forecast line
- render 3D flood map for `+20cm / +50cm / +100cm`
- show provider, support tier, forecast mode, and delay notes
- switch to `Multi-city compare` to compare Honolulu / Boston / New York / Jakarta / Amsterdam in one view

## GIS logic
- DEM cells are treated as land only when `elevation > 0 m`.
- Flood scenarios are built from `0 < DEM <= scenario_water_level`.
- Only coast-connected components are kept.
- Small polygons are filtered, then geometries are cleaned before exporting GeoJSON/GPKG.
- For steep locations like Honolulu, `+50 cm` and even `+1 m` can still appear as narrow coastal strips. That is expected for a simple DEM-threshold workflow.

## Rebuild GIS outputs after code update
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.run_pipeline \
  --csv data/honolulu_hourly.csv \
  --dem data/honolulu_dem.tif \
  --value-col sea_level \
  --time-col timestamp \
  --model-type axial_lstm \
  --reuse-model \
  --horizon 6 \
  --out Backend/sea_level_risk/outputs
```

Then regenerate summary maps:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.postprocess --out Backend/sea_level_risk/outputs
```

Create a QGIS-ready package from the cleaned outputs:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.qgis.prepare_qgis_package \
  --city honolulu \
  --dem data/honolulu_dem.tif \
  --realtime-dir Backend/sea_level_risk/outputs \
  --out-root Backend/sea_level_risk/outputs/qgis_packages
```

Each generated package also contains `CREATE_TEMPLATE_IN_QGIS_CONSOLE.py`.
Open QGIS Python Console and run that package-local script to create `anti_flood_template.qgz`.

## Warm up city assets (Boston / New York / others)
This will fetch realtime data, auto-download DEM if missing, generate scenario outputs, and build a QGIS package when possible:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.prepare_city_assets \
  --cities boston,newyork \
  --horizon 6 \
  --hours-back 96 \
  --auto-dem
```

Examples:
- `--cities jakarta --auto-dem`
- `--cities boston,newyork,jakarta --auto-dem`

Note:
- `amsterdam` may still build only delayed/proxy outputs because the upstream IOC feed is often stale.

## Files
- `Backend/sea_level_risk/realtime_api.py`: realtime API
- `Backend/sea_level_risk/dashboard_app.py`: 3D dashboard
- `Backend/sea_level_risk/city_registry.json`: city registry with provider/support metadata
- `Backend/sea_level_risk/data_providers.py`: NOAA + IOC fetch layer
- `Backend/sea_level_risk/forecast_baselines.py`: tide-aware baseline forecast
- `Backend/sea_level_risk/dem_provider.py`: auto DEM from Copernicus

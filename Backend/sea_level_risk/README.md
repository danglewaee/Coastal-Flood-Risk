# Coastal Flood Risk

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

## Train city-specific deep-learning models
Train one city at a time into `Backend/sea_level_risk/outputs/models/<city>/`.

Boston example (automatic NOAA hourly history download):
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.train_city_model \
  --city boston \
  --begin 20100101 \
  --end 20251231 \
  --model-type axial_lstm \
  --feature-mode multivariate_v1
```

New York example:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.train_city_model \
  --city newyork \
  --begin 20100101 \
  --end 20251231 \
  --model-type axial_lstm \
  --feature-mode multivariate_v1
```

If a city does not have automatic hourly history download implemented, provide your own hourly CSV:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.train_city_model \
  --city jakarta \
  --csv data/jakarta_hourly.csv \
  --time-col timestamp \
  --value-col sea_level \
  --model-type axial_lstm \
  --feature-mode multivariate_v1
```

Important:
- The local files `data/boston.csv`, `data/newyork.csv`, `data/jakarta.csv`, and `data/amsterdam.csv` are coarse monthly trend series, not hourly gauge histories. Do not use them for the current short-horizon forecast task.
- The realtime API will automatically prefer `Backend/sea_level_risk/outputs/models/<city>/` when a city-specific model exists. Otherwise it falls back to the tide-aware baseline.
- `multivariate_v1` uses derived sequence features from the hourly water-level signal: level, first-difference, rolling statistics, hour-of-day cycle, and semidiurnal tidal cycle encodings.
- Trained models now store validation-residual uncertainty calibration so the API can return `P10 / P50 / P90` forecast bands.
- Legacy city models remain usable. They will report `feature_mode = univariate_v0` until they are retrained with `--feature-mode multivariate_v1`.

## Forecast accuracy backtesting (Phase Accuracy-1)
Run rolling-origin backtests to compare each city model against the tide-persistence baseline before retraining or adding new features.

Boston example (last 30 days, 6-hour horizon, one forecast every 6 hours):
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.backtest \
  --cities boston \
  --horizon 6 \
  --step-hours 6 \
  --eval-window-hours 720
```

All NOAA cities:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.backtest --all-noaa --horizon 6 --step-hours 6 --eval-window-hours 720
```

Artifacts are written to `Backend/sea_level_risk/outputs/backtests/<city>/`:
- `summary.json`: aggregate metrics for each forecaster
- `horizon_metrics.csv`: metric-by-horizon table
- `window_forecasts.csv`: every rolling forecast window, suitable for custom plotting

Forecaster selection mirrors the runtime service:
- cities with `outputs/models/<city>/` use the city-specific model
- `honolulu` falls back to the legacy global model under `Backend/sea_level_risk/outputs/`
- every selected city can still be benchmarked against the tide-persistence baseline

Current backtest metrics include:
- overall `MAE / RMSE / bias`
- `high_water_mae_m` and `high_water_rmse_m` on the top-decile observed water levels
- `peak_level_mae_m` and `peak_timing_mae_h`
- `P10-P90` interval coverage and mean interval width

Recalibrate uncertainty bands from rolling backtest residuals:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.recalibrate_uncertainty --all-noaa --horizon 6 --step-hours 6 --eval-window-hours 1440
```

This updates each model metadata file in place so runtime forecasts can use horizon-specific residual quantiles instead of a single shared residual band.

## Realtime API (multi-city, provider-aware)
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.realtime_api \
  --model Backend/sea_level_risk/outputs/sea_level_axial_lstm.keras \
  --metadata Backend/sea_level_risk/outputs/metadata.json \
  --models-root Backend/sea_level_risk/outputs/models \
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
- `GET /realtime/briefing?city=boston&scenario=plus_50cm&horizon=6&hours_back=96&auto_dem=1`
- `GET /realtime/forecast?provider=noaa&station=1612340&horizon=6&hours_back=96&auto_dem=1`

Notes:
- `city_registry.json` now stores provider/station metadata, support tier, proxy mode, forecast mode, and DEM hints per city.
- `city_registry.json` also stores prototype operational alert thresholds and notes per city.
- `honolulu` uses the trained deep-learning model.
- Any city with a trained model under `Backend/sea_level_risk/outputs/models/<city>/` will use that model automatically.
- Cities without a city-specific model fall back to the tide-aware baseline.
- `forecast_values_m` now represents the calibrated `P50` trajectory.
- `forecast_quantiles` and `forecast[*].p10_m / p50_m / p90_m` expose probabilistic outputs.
- NOAA realtime fetch uses `water_level`, not `hourly_height`.
- IOC realtime fetch parses the public HTML table feed and resamples it to hourly series.
- `amsterdam` is implemented as `Amsterdam-region proxy (Hoek van Holland)`. Treat it as delayed regional proxy mode, not a direct Amsterdam city gauge.
- If a city has no local DEM, `auto_dem=1` will fetch Copernicus DEM by NOAA station metadata or by city `lat/lon` from the registry.
- When a city has `lat/lon`, the service now prefers a city-clipped DEM cache under `data/dem_city_cache/` instead of using the full 1x1 Copernicus tile for every request. This keeps map generation faster for cities such as Miami.

## Supported city modes
- `honolulu`: official NOAA realtime + local deep model
- `boston`: official NOAA realtime + city model when trained, otherwise tide-aware baseline
- `newyork`: official NOAA realtime + city model when trained, otherwise tide-aware baseline
- `jakarta`: experimental IOC realtime + city model when trained on hourly data, otherwise tide-aware baseline
- `amsterdam`: delayed IOC coastal proxy + city model when trained on hourly data, otherwise tide-aware baseline

## Dashboard (interactive)
Run the API first, then run dashboard:
```powershell
Backend/.venv311/Scripts/streamlit run Backend/sea_level_risk/dashboard_app.py
```

Dashboard features:
- choose a coastal city
- fetch provider-aware realtime data
- render operational summary, alert level, and recommended actions for the selected scenario
- render prioritized flood hotspots and scenario-level impact summaries
- render a practical 2D flood polygon map on a real basemap
- render `P10 / P50 / P90` forecast trajectories when probabilistic outputs are available
- render 3D flood map for `+20cm / +50cm / +100cm`
- show provider, support tier, forecast mode, and delay notes
- download a city/scenario operational briefing as Markdown
- switch to `Multi-city compare` to compare Honolulu / Boston / New York / Jakarta / Amsterdam in one view

## Presentation-ready app
For demo / report / advisor presentation, use the separate presentation app:
```powershell
Backend/.venv311/Scripts/streamlit run Backend/sea_level_risk/presentation_app.py --server.port 8602
```

One-click launcher:
```powershell
powershell -ExecutionPolicy Bypass -File Backend/sea_level_risk/start_presentation_stack.ps1
```

Stop both services:
```powershell
powershell -ExecutionPolicy Bypass -File Backend/sea_level_risk/stop_presentation_stack.ps1
```

Presentation app layout:
- city spotlight
- operational summary with alert/confidence and downloadable briefing
- prioritized hotspot summary for the selected scenario
- 2D flood map on a basemap
- forecast trajectory
- cross-city compare table
- executive takeaway cards
- methodology and limitations section

## GIS logic
- DEM cells are treated as land only when `elevation > 0 m`.
- Flood scenarios are built from `0 < DEM <= scenario_water_level`.
- Only coast-connected components are kept.
- Small polygons are filtered, then geometries are cleaned before exporting GeoJSON/GPKG.
- Hotspots are ranked across scenario polygons using a weighted score based on scenario severity, city-level flood ratio, and polygon area.
- For steep locations like Honolulu, `+50 cm` and even `+1 m` can still appear as narrow coastal strips. That is expected for a simple DEM-threshold workflow.

## Exposure layers (Phase 4)
The repo now ships with a tracked exposure registry:
- `Backend/sea_level_risk/exposure_registry.json`

Default expected local paths:
- `data/exposure/<city>/roads.geojson`
- `data/exposure/<city>/critical_facilities.geojson`

Download real exposure layers from OpenStreetMap Overpass:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.download_exposure_layers --cities boston newyork honolulu jakarta amsterdam
```

Download every city in the registry:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.download_exposure_layers --all-known
```

Notes:
- The downloader creates local GeoJSON layers under `data/exposure/`.
- Roads are exported as line features.
- Critical facilities are exported as point features from OSM `hospital`, `clinic`, `police`, and `fire_station` tags.
- Scenario payloads now include `impact_summaries[*].exposure_summary` for any registered layer present on disk.
- Exposure summaries report intersections plus geometry-aware metrics:
  - polygon area in `affected_area_m2`
  - line length in `affected_length_m`
  - point counts in `affected_point_count`

The current implementation is still intentionally pragmatic:
- it computes direct scenario-layer intersections
- it does not yet perform network accessibility analysis
- it does not yet estimate exposed population or service capacity

## Hydrodynamic integration (Boston PoC)
If you run a 2D hydrodynamic model (e.g., HEC-RAS 2D) and export a depth grid GeoTIFF,
you can replace the DEM-threshold flood polygons with model-based polygons.

Workflow:
1. Run the hydrodynamic model for the scenario (e.g., `plus_50cm`).
2. Export depth grid to GeoTIFF from the model (RAS Mapper export).
3. Postprocess depth raster into flood polygons:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.hydro_postprocess ^
  --city boston ^
  --scenario plus_50cm ^
  --depth data/hydro/boston/plus_50cm_depth.tif ^
  --dem data/dem_cache/Copernicus_DSM_COG_10_N42_00_W072_00_DEM.tif
```

Notes:
- Depth raster and DEM must be aligned (same grid/CRS). Otherwise flood ratio will be omitted.
- The output is written to `Backend/sea_level_risk/outputs/realtime/boston/flood_plus_50cm.geojson`
- When a scenario GeoJSON has `processing_mode = "hydro_model"`, the realtime API will prefer it over DEM-threshold output.

Detailed HEC-RAS steps:
- `Backend/sea_level_risk/docs/HEC_RAS_BOSTON.md`

Quick alignment check:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.check_alignment ^
  --dem data/dem_cache/Copernicus_DSM_COG_10_N42_00_W072_00_DEM.tif ^
  --depth data/hydro/boston/plus_50cm_depth.tif
```

Batch postprocess (all 3 scenarios if present):
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.hydro_batch ^
  --city boston ^
  --depth-dir data/hydro/boston ^
  --dem data/dem_cache/Copernicus_DSM_COG_10_N42_00_W072_00_DEM.tif
```

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
- `Backend/sea_level_risk/model_registry.py`: city-specific model discovery
- `Backend/sea_level_risk/train_city_model.py`: train one city model into `outputs/models/<city>/`
- `Backend/sea_level_risk/dashboard_app.py`: 3D dashboard
- `Backend/sea_level_risk/presentation_app.py`: presentation-ready demo app
- `Backend/sea_level_risk/city_registry.json`: city registry with provider/support metadata
- `Backend/sea_level_risk/data_providers.py`: NOAA + IOC fetch layer
- `Backend/sea_level_risk/forecast_baselines.py`: tide-aware baseline forecast
- `Backend/sea_level_risk/dem_provider.py`: auto DEM from Copernicus

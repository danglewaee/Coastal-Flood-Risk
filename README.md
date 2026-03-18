# Coastal Flood Risk

Realtime sea-level forecasting + GIS flood-risk mapping project.

## What this repo includes
- `Backend/sea_level_risk/`: core pipeline and services
  - one-step forecasting models (`lstm`, `temporal_cnn`, `axial_lstm`)
  - realtime API (`/realtime/forecast`, `/realtime/hotspots`, `/realtime/cities`)
  - GIS inundation mapping from DEM
  - hotspot priority scoring
  - Streamlit dashboard (forecast + 3D map)
- `Backend/sea_level_risk/qgis/`: QGIS standardization toolkit
  - package builder
  - template generator
  - layer styles (`.qml`)

## Quickstart (Windows)

### 1) Setup Python environment
```powershell
powershell -ExecutionPolicy Bypass -File Backend/setup_py311_env.ps1
Backend/.venv311/Scripts/python -m pip install -r Backend/requirements-ml.txt
```

### 2) Start realtime API
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.realtime_api --host 127.0.0.1 --port 8100
```

### 3) Start dashboard
```powershell
Backend/.venv311/Scripts/python -m streamlit run Backend/sea_level_risk/dashboard_app.py --server.port 8601
```

Open:
- Dashboard: `http://localhost:8601`
- API health: `http://127.0.0.1:8100/health`

## API endpoints
- `GET /health`
- `GET /realtime/cities`
- `GET /realtime/forecast?city=honolulu&horizon=6&hours_back=96&auto_dem=1`
- `GET /realtime/hotspots?city=honolulu&limit=10`

## QGIS workflow
1. Build package:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.qgis.prepare_qgis_package --city honolulu --dem data/honolulu_dem.tif --realtime-dir Backend/sea_level_risk/outputs/realtime/honolulu --out-root Backend/sea_level_risk/outputs/qgis_packages
```
2. Generate template project in QGIS Python Console:
```python
import sys
sys.path.append(r"D:\CODE\Projects\Sea_Level_Rise - New ver")
from Backend.sea_level_risk.qgis.create_qgis_template import create_project
create_project(r"D:\CODE\Projects\Sea_Level_Rise - New ver\Backend\sea_level_risk\outputs\qgis_packages\<package_folder>")
```

See:
- `Backend/sea_level_risk/qgis/QGIS_STANDARD.md`
- `Backend/sea_level_risk/qgis/QGIS_TEMPLATE_GUIDE.md`

## Notes
- Large data/model artifacts are intentionally ignored by Git.
- This repo currently tracks the backend core and GIS tooling first.

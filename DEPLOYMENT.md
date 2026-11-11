# Deploying The Interactive Explorer

The React/Leaflet dashboard requires the Python API and GIS assets. Deploying only the static frontend will not provide flood analysis.

## One Origin (Recommended)

1. Install Python 3.11 and Node 22.12+ or 24 on the host.
2. Build the frontend from `Frontend/react-app` with `npm ci` and `npm run build`.
3. From the repository root, install `Backend/requirements-explorer.txt`. For observation refresh with ML, also install `Backend/requirements-ml.txt`.
4. Provision the city GIS outputs, DEM clips, exposure layers and model assets separately; they are not stored in Git. Use a persistent data volume for generated outputs and forecast snapshots.
5. Start Waitress from the repository root:

```powershell
Backend\.venv311\Scripts\waitress-serve --call --listen=127.0.0.1:8100 Backend.sea_level_risk.explorer_api:create_runtime_app
```

Place the server behind an HTTPS reverse proxy. Flask serves `Frontend/react-app/dist` and supports React page URLs; `/api` stays on the same origin. Use a proxy timeout long enough for an explicit observation refresh. Saved-map browsing does not require TensorFlow or provider access.

For a local production-build check:

```powershell
Backend\.venv311\Scripts\python -m Backend.sea_level_risk.explorer_api --port 8100
```

Open `http://127.0.0.1:8100` after building. Flask's development server is for local use; use Waitress for deployment.

## Separate Frontend

Netlify/Vercel configuration in this repo builds `Frontend/react-app/dist`. Set `VITE_API_BASE_URL` to the public Python service's **`/api` URL** at build time and configure explicit CORS origins on that service. The original `realtime_api` already registers the explorer routes and Flask-CORS; the lightweight `explorer_api` defaults to same-origin access.

Never point a hosted frontend to `localhost` or `127.0.0.1`: that would address the visitor's computer. A backend host with the GIS/ML dependencies and data volume is required. No hosted backend has been provisioned by this change.

## API

```text
GET  /health
GET  /api/catalog
GET  /api/cities/boston/analysis?scenario=plus_50cm
GET  /api/cities/boston/components/1?scenario=plus_50cm&version=<analysis-version>
GET  /api/cities/boston/evidence
POST /api/cities/boston/refresh  {"scenario":"plus_50cm","horizon":6}
```

Only registered cities and scenario identifiers are accepted. Selection includes the analysis version so a stale polygon cannot silently return metrics from a newer footprint.

## Verification

```powershell
Backend\.venv311\Scripts\python -m unittest discover -s Backend/tests -v
cd Frontend/react-app
npm run build
npm test
```

Browser tests use Microsoft Edge and a running API/Vite pair. Set `E2E_BASE_URL` to test the built frontend through Flask instead. External basemap tiles require internet; polygons and exposure calculations come from local assets.

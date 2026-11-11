# Coastal Flood Explorer

React 19 + Leaflet 1.9 + React Leaflet 5. Vite serves development on port 5173 and proxies `/api` to Flask on port 8100.

```powershell
npm ci
npm start
npm run build
```

Start the backend from the repository root using `python -m Backend.sea_level_risk.explorer_api --port 8100`. Build output is `dist/`. Set `VITE_API_BASE_URL` for a separately hosted API; omit it for same-origin production serving.

Pages share a persistent map and store city/scenario/selection in the URL. Analysis and inspection requests are cancelled on selection changes to prevent stale responses. Saved analyses are reused; live observations refresh only on explicit request.

Browser checks with the API and frontend running, using installed Microsoft Edge:

```powershell
npm test
```

See the root [README](../../README.md) and [deployment guide](../../DEPLOYMENT.md) for assets, provenance, limitations and production hosting. The older global-altimetry static showcase is no longer the frontend entry point.

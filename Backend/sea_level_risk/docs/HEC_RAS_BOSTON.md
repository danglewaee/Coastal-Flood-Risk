# HEC-RAS 2D Boston PoC (Depth Raster -> Flood Polygon)

This guide creates a depth raster in HEC-RAS 2D and plugs it into the existing
pipeline using `hydro_postprocess.py`.

## 0) Inputs you need
- Terrain (DEM), ideally 1-3 m.
- Bathymetry for Boston Harbor / main channels (optional but strongly recommended).
- Boundary conditions:
  - Tidal stage at the open boundary.
  - River inflow (if modeling upstream discharge).

## 1) Prepare terrain
1. Build a terrain in HEC-RAS (RAS Mapper).
2. Confirm CRS is consistent (projected CRS is preferred).
3. If bathymetry is available, merge into the terrain.

## 2) Build 2D geometry
1. Create a 2D Flow Area covering the harbor + nearshore.
2. Set mesh resolution (e.g., 30-50 m for PoC, finer near channels if needed).
3. Add breaklines along shorelines / channels if you have bathymetry.

## 3) Boundary conditions
1. Add a stage boundary at the open-coast edge (tide).
2. Add upstream inflow boundary if modeling rivers.
3. Set simulation time window and warm-up period.

## 4) Run unsteady flow
1. Run the unsteady simulation.
2. Confirm stable run (no major instabilities).

## 5) Export depth raster
1. Open RAS Mapper.
2. Select `Depth` results.
3. Export raster to GeoTIFF.
4. Name file as:
   - `data/hydro/boston/plus_20cm_depth.tif`
   - `data/hydro/boston/plus_50cm_depth.tif`
   - `data/hydro/boston/plus_100cm_depth.tif`

## 6) Convert depth raster to flood polygon
Run:
```powershell
Backend/.venv311/Scripts/python -m Backend.sea_level_risk.hydro_postprocess ^
  --city boston ^
  --scenario plus_50cm ^
  --depth data/hydro/boston/plus_50cm_depth.tif ^
  --dem data/dem_cache/Copernicus_DSM_COG_10_N42_00_W072_00_DEM.tif
```

This will write:
- `Backend/sea_level_risk/outputs/realtime/boston/flood_plus_50cm.geojson`
- `Backend/sea_level_risk/outputs/realtime/boston/flood_plus_50cm.meta.json`

When `processing_mode = "hydro_model"` in the meta file, the realtime API will
prefer this output over the DEM-threshold method.

## 7) Validate
- Check the output GeoJSON in QGIS.
- Compare flood extents vs. known high-tide events.
- Iterate mesh/boundary conditions if needed.

## Notes
- Depth raster and DEM should be aligned (same CRS, resolution, extent).
- Use `--depth-threshold 0.01` or 0.02 m if you see noisy speckles.

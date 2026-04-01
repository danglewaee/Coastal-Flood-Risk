import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.prepared import prep


EQUAL_AREA_EPSG = 6933
DEFAULT_MIN_LAND_ELEVATION_M = 0.0
DEFAULT_MIN_COMPONENT_AREA_M2 = 2500.0
DEFAULT_SMOOTH_TOLERANCE_M = 10.0


def _empty_flood_gdf(crs, predicted_level_m: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "predicted_level_m": [],
            "component_id": [],
            "component_area_m2": [],
            "coastal_connected": [],
        },
        geometry=[],
        crs=crs,
    )


def _compute_flood_area_m2(flood_gdf: gpd.GeoDataFrame) -> float:
    if flood_gdf.empty:
        return 0.0
    if "component_area_m2" in flood_gdf.columns and not flood_gdf["component_area_m2"].isna().all():
        return float(flood_gdf["component_area_m2"].sum())
    return float(flood_gdf.to_crs(epsg=EQUAL_AREA_EPSG).geometry.area.sum())


def _mask_to_geometries(mask: np.ndarray, transform) -> list:
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.size == 0 or int(mask.sum()) == 0:
        return []

    geoms = []
    mask_bool = mask.astype(bool)
    for geom, value in shapes(mask, mask=mask_bool, transform=transform, connectivity=8):
        if value == 1:
            geoms.append(shape(geom))
    return geoms


def _select_open_background(background_geoms: list, raster_bounds) -> list:
    if not background_geoms:
        return []

    bounds_geom = box(*raster_bounds)
    open_background = []
    for geom in background_geoms:
        if geom.is_empty:
            continue
        if geom.intersects(bounds_geom.boundary):
            open_background.append(geom)
    return open_background


def _clean_flood_geometries(
    geoms: list,
    predicted_level_m: float,
    crs,
    min_component_area_m2: float,
    smooth_tolerance_m: float,
) -> gpd.GeoDataFrame:
    if not geoms:
        return _empty_flood_gdf(crs, predicted_level_m)

    gdf = gpd.GeoDataFrame(
        {"predicted_level_m": [predicted_level_m] * len(geoms)},
        geometry=geoms,
        crs=crs,
    )
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return _empty_flood_gdf(crs, predicted_level_m)

    gdf_metric = gdf.to_crs(epsg=EQUAL_AREA_EPSG)
    gdf_metric["geometry"] = gdf_metric.geometry.buffer(0)
    gdf_metric = gdf_metric[gdf_metric.geometry.notna() & ~gdf_metric.geometry.is_empty].copy()
    if gdf_metric.empty:
        return _empty_flood_gdf(crs, predicted_level_m)

    if min_component_area_m2 > 0:
        gdf_metric = gdf_metric[gdf_metric.geometry.area >= min_component_area_m2].copy()
    if gdf_metric.empty:
        return _empty_flood_gdf(crs, predicted_level_m)

    merged = unary_union(list(gdf_metric.geometry))
    merged_series = gpd.GeoSeries([merged], crs=f"EPSG:{EQUAL_AREA_EPSG}")
    merged_series = merged_series.explode(index_parts=False).reset_index(drop=True)

    clean = gpd.GeoDataFrame(geometry=merged_series, crs=f"EPSG:{EQUAL_AREA_EPSG}")
    clean = clean[clean.geometry.notna() & ~clean.geometry.is_empty].copy()
    if clean.empty:
        return _empty_flood_gdf(crs, predicted_level_m)

    if smooth_tolerance_m > 0:
        clean["geometry"] = clean.geometry.simplify(smooth_tolerance_m, preserve_topology=True).buffer(0)
        clean = clean[clean.geometry.notna() & ~clean.geometry.is_empty].copy()
        if clean.empty:
            return _empty_flood_gdf(crs, predicted_level_m)

    clean["component_area_m2"] = clean.geometry.area.astype(float)
    if min_component_area_m2 > 0:
        clean = clean[clean["component_area_m2"] >= min_component_area_m2].copy()
    if clean.empty:
        return _empty_flood_gdf(crs, predicted_level_m)

    clean = clean.sort_values("component_area_m2", ascending=False).reset_index(drop=True)
    clean["predicted_level_m"] = predicted_level_m
    clean["component_id"] = np.arange(1, len(clean) + 1, dtype=int)
    clean["coastal_connected"] = True

    return clean.to_crs(crs)


def dem_to_flood_polygon(
    dem_path: str,
    predicted_level_m: float,
    out_geojson: str,
    crs: Optional[str] = None,
    min_land_elevation_m: float = DEFAULT_MIN_LAND_ELEVATION_M,
    min_component_area_m2: float = DEFAULT_MIN_COMPONENT_AREA_M2,
    smooth_tolerance_m: float = DEFAULT_SMOOTH_TOLERANCE_M,
) -> dict:
    out_path = Path(out_geojson)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=True)
        nodata_mask = np.ma.getmaskarray(dem)
        dem_data = np.asarray(dem.filled(np.nan), dtype=np.float32)

        valid_mask = np.asarray(~nodata_mask, dtype=bool)
        land_mask = valid_mask & np.isfinite(dem_data) & (dem_data > float(min_land_elevation_m))
        flood_candidate_mask = land_mask & (dem_data <= float(predicted_level_m))
        background_mask = ~land_mask

        flood_candidate_geoms = _mask_to_geometries(flood_candidate_mask, src.transform)
        background_geoms = _mask_to_geometries(background_mask, src.transform)
        open_background_geoms = _select_open_background(background_geoms, src.bounds)

        if not flood_candidate_geoms or not open_background_geoms:
            flood_gdf = _empty_flood_gdf(src.crs, predicted_level_m)
            coastal_connected_mask = np.zeros_like(flood_candidate_mask, dtype=bool)
        else:
            open_background_union = unary_union(open_background_geoms)
            prepared_background = prep(open_background_union)
            coastal_geoms = [geom for geom in flood_candidate_geoms if prepared_background.intersects(geom)]

            flood_gdf = _clean_flood_geometries(
                geoms=coastal_geoms,
                predicted_level_m=predicted_level_m,
                crs=src.crs,
                min_component_area_m2=min_component_area_m2,
                smooth_tolerance_m=smooth_tolerance_m,
            )

            if flood_gdf.empty:
                coastal_connected_mask = np.zeros_like(flood_candidate_mask, dtype=bool)
            else:
                coastal_connected_mask = rasterize(
                    ((geom, 1) for geom in flood_gdf.geometry),
                    out_shape=flood_candidate_mask.shape,
                    transform=src.transform,
                    fill=0,
                    dtype=np.uint8,
                ).astype(bool)
                coastal_connected_mask &= flood_candidate_mask

        if crs and not flood_gdf.empty:
            flood_gdf = flood_gdf.to_crs(crs)
        elif crs:
            flood_gdf = flood_gdf.set_crs(src.crs, allow_override=True).to_crs(crs)

        flood_gdf.to_file(out_path, driver="GeoJSON")

        land_pixel_count = int(np.count_nonzero(land_mask))
        flood_pixel_count = int(np.count_nonzero(coastal_connected_mask))
        candidate_pixel_count = int(np.count_nonzero(flood_candidate_mask))
        flood_ratio = float(flood_pixel_count / land_pixel_count) if land_pixel_count > 0 else 0.0
        flood_area_m2 = _compute_flood_area_m2(flood_gdf)

    return {
        "predicted_level_m": predicted_level_m,
        "processing_mode": "coastal_connected_threshold",
        "min_land_elevation_m": float(min_land_elevation_m),
        "min_component_area_m2": float(min_component_area_m2),
        "smooth_tolerance_m": float(smooth_tolerance_m),
        "flood_pixels": flood_pixel_count,
        "candidate_flood_pixels": candidate_pixel_count,
        "land_pixels": land_pixel_count,
        "valid_pixels": int(np.count_nonzero(valid_mask)),
        "component_count": int(len(flood_gdf)),
        "flood_ratio": flood_ratio,
        "flood_area_m2": flood_area_m2,
        "out_geojson": str(out_path),
    }


def depth_raster_to_flood_polygon(
    depth_path: str,
    out_geojson: str,
    dem_path: Optional[str] = None,
    depth_threshold_m: float = 0.01,
    min_land_elevation_m: float = DEFAULT_MIN_LAND_ELEVATION_M,
    min_component_area_m2: float = DEFAULT_MIN_COMPONENT_AREA_M2,
    smooth_tolerance_m: float = DEFAULT_SMOOTH_TOLERANCE_M,
) -> dict:
    out_path = Path(out_geojson)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    land_pixel_count = None
    flood_pixel_count = 0
    flood_ratio = 0.0

    with rasterio.open(depth_path) as src:
        depth = src.read(1, masked=True)
        nodata_mask = np.ma.getmaskarray(depth)
        depth_data = np.asarray(depth.filled(np.nan), dtype=np.float32)

        valid_mask = np.asarray(~nodata_mask, dtype=bool)
        flood_mask = valid_mask & np.isfinite(depth_data) & (depth_data > float(depth_threshold_m))

        flood_pixel_count = int(np.count_nonzero(flood_mask))
        flood_geoms = _mask_to_geometries(flood_mask, src.transform)
        flood_gdf = _clean_flood_geometries(
            geoms=flood_geoms,
            predicted_level_m=float(depth_threshold_m),
            crs=src.crs,
            min_component_area_m2=min_component_area_m2,
            smooth_tolerance_m=smooth_tolerance_m,
        )

        if dem_path and Path(dem_path).exists():
            with rasterio.open(dem_path) as dem_src:
                if (
                    dem_src.width == src.width
                    and dem_src.height == src.height
                    and dem_src.transform == src.transform
                    and dem_src.crs == src.crs
                ):
                    dem = dem_src.read(1, masked=True)
                    dem_data = np.asarray(dem.filled(np.nan), dtype=np.float32)
                    dem_valid = np.asarray(~np.ma.getmaskarray(dem), dtype=bool)
                    land_mask = dem_valid & np.isfinite(dem_data) & (dem_data > float(min_land_elevation_m))
                    land_pixel_count = int(np.count_nonzero(land_mask))
                    flood_land_pixels = int(np.count_nonzero(flood_mask & land_mask))
                    if land_pixel_count > 0:
                        flood_ratio = float(flood_land_pixels / land_pixel_count)

        flood_gdf.to_file(out_path, driver="GeoJSON")
        flood_area_m2 = _compute_flood_area_m2(flood_gdf)

    return {
        "processing_mode": "hydro_depth_threshold",
        "depth_threshold_m": float(depth_threshold_m),
        "min_land_elevation_m": float(min_land_elevation_m),
        "min_component_area_m2": float(min_component_area_m2),
        "smooth_tolerance_m": float(smooth_tolerance_m),
        "land_pixels": int(land_pixel_count) if land_pixel_count is not None else None,
        "flood_pixels": int(flood_pixel_count),
        "flood_ratio": float(flood_ratio),
        "component_count": int(len(flood_gdf)) if flood_gdf is not None else 0,
        "flood_area_m2": float(flood_area_m2),
        "out_geojson": str(out_path),
    }


def compute_exposure(flood_geojson: str, layer_path: str, layer_name: str = "layer") -> dict:
    flood = gpd.read_file(flood_geojson)
    layer = gpd.read_file(layer_path)

    if flood.empty or layer.empty:
        return {
            "layer": layer_name,
            "intersections": 0,
            "affected_area_m2": 0.0,
            "affected_length_m": 0.0,
            "affected_point_count": 0,
        }

    if flood.crs != layer.crs:
        layer = layer.to_crs(flood.crs)

    intersection = gpd.overlay(layer, flood, how="intersection", keep_geom_type=False)
    if intersection.empty:
        return {
            "layer": layer_name,
            "intersections": 0,
            "affected_area_m2": 0.0,
            "affected_length_m": 0.0,
            "affected_point_count": 0,
        }

    metric_gdf = intersection.to_crs(epsg=EQUAL_AREA_EPSG)
    geom_types = metric_gdf.geometry.geom_type.str.lower()
    polygon_mask = geom_types.isin(["polygon", "multipolygon"])
    line_mask = geom_types.isin(["linestring", "multilinestring"])
    point_mask = geom_types.isin(["point", "multipoint"])

    area = float(metric_gdf.loc[polygon_mask, "geometry"].area.sum()) if polygon_mask.any() else 0.0
    length = float(metric_gdf.loc[line_mask, "geometry"].length.sum()) if line_mask.any() else 0.0
    point_count = int(point_mask.sum())

    source_geom_types = sorted({str(item).lower() for item in layer.geometry.geom_type.dropna().unique().tolist()})

    return {
        "layer": layer_name,
        "intersections": int(len(intersection)),
        "affected_area_m2": area,
        "affected_length_m": length,
        "affected_point_count": point_count,
        "source_geometry_types": source_geom_types,
    }


def save_summary(path: str, summary: dict) -> None:
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")

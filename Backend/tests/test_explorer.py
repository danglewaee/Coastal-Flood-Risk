import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, MultiPoint, Point, box

from Backend.sea_level_risk.explorer_api import create_app
from Backend.sea_level_risk.explorer_service import ExplorerService, REPO_ROOT
from Backend.sea_level_risk.gis import compute_exposure_frames
from Backend.sea_level_risk.priority import rank_hotspots_from_scenarios


class ExposureTests(unittest.TestCase):
    def test_overlapping_flood_polygons_do_not_double_count_sites(self):
        flood = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2), box(1, 0, 3, 2)], crs=4326)
        sites = gpd.GeoDataFrame(geometry=[Point(1.5, 1), Point(4, 4)], crs=4326)
        result = compute_exposure_frames(flood, sites)
        self.assertEqual(result['affected_point_count'], 1)

    def test_multipoints_count_individual_facilities(self):
        flood = gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)], crs=4326)
        sites = gpd.GeoDataFrame(geometry=[MultiPoint([(1, 1), (2, 2)])], crs=4326)
        self.assertEqual(compute_exposure_frames(flood, sites)['affected_point_count'], 2)

    def test_population_is_weighted_by_intersection_area(self):
        flood = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=6933)
        people = gpd.GeoDataFrame({'people': [100]}, geometry=[box(0, 0, 2, 1)], crs=6933)
        result = compute_exposure_frames(flood, people, value_field='people', weight_by_area=True)
        self.assertAlmostEqual(result['affected_weighted_value'], 50)

    def test_road_length_uses_local_metric_crs(self):
        flood = gpd.GeoDataFrame(geometry=[box(320000, 4689990, 320100, 4690010)], crs=32619)
        roads = gpd.GeoDataFrame(geometry=[LineString([(319900, 4690000), (320200, 4690000)])], crs=32619)
        self.assertAlmostEqual(compute_exposure_frames(flood, roads)['affected_length_m'], 100, delta=0.05)

    def test_missing_crs_rejected(self):
        flood = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])
        sites = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs=4326)
        with self.assertRaises(ValueError):
            compute_exposure_frames(flood, sites)

    def test_empty_inundation_has_no_hotspots(self):
        self.assertTrue(rank_hotspots_from_scenarios([]).empty)


class ExplorerApiTests(unittest.TestCase):
    def setUp(self):
        temporary_root = REPO_ROOT / '.cache/tests'
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temp.name)
        backend = self.root / 'Backend/sea_level_risk'
        backend.mkdir(parents=True)
        (backend / 'city_registry.json').write_text(json.dumps({'boston': {'display_name': 'Boston', 'lat': 42.36, 'lon': -71.06}}))
        exposure = self.root / 'data/sites.geojson'
        exposure.parent.mkdir()
        gpd.GeoDataFrame(geometry=[Point(-71.01, 42.01)], crs=4326).to_file(exposure, driver='GeoJSON')
        (backend / 'exposure_registry.json').write_text(json.dumps({'boston': [
            {'name': 'sites', 'path': 'data/sites.geojson', 'metric': 'points', 'category': 'healthcare'},
            {'name': 'roads', 'path': 'data/missing.geojson', 'metric': 'length_m', 'category': 'transport'},
        ]}))
        outputs = backend / 'outputs/realtime/boston'
        outputs.mkdir(parents=True)
        for name in ('plus_20cm', 'plus_50cm', 'plus_100cm'):
            gpd.GeoDataFrame(geometry=[box(-71.02, 42, -71, 42.02)], crs=4326).to_file(outputs / f'flood_{name}.geojson', driver='GeoJSON')
        self.client = create_app(self.root).test_client()

    def tearDown(self):
        self.temp.cleanup()

    def test_initial_analysis_uses_real_assets_and_reports_missing_layers(self):
        response = self.client.get('/api/cities/boston/analysis')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['metrics']['sites'], 1)
        self.assertIsNone(data['metrics']['roads_m'])
        self.assertIsNone(data['observation_utc'])
        self.assertEqual(data['exposure']['layers'][1]['status'], 'missing')
        self.assertEqual(data['vertical_alignment'], 'unverified')
        self.assertEqual(data['forecast'], [])

    def test_component_inspection_and_stale_version(self):
        data = self.client.get('/api/cities/boston/analysis').get_json()
        response = self.client.get(f"/api/cities/boston/components/1?version={data['version']}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['exposure']['metrics']['sites'], 1)
        self.assertEqual(len(response.get_json()['comparison']), 3)
        self.assertEqual(self.client.get('/api/cities/boston/components/1?version=stale').status_code, 409)

    def test_unknown_city_and_scenario(self):
        self.assertEqual(self.client.get('/api/cities/unknown/analysis').status_code, 404)
        self.assertEqual(self.client.get('/api/cities/boston/analysis?scenario=unsafe').status_code, 400)
        self.assertEqual(self.client.get('/api/cities/boston/components/9999').status_code, 404)

    def test_no_path_traversal(self):
        service = ExplorerService(self.root)
        with self.assertRaises(ValueError):
            service._path('../outside.geojson')

    def test_live_refresh_validates_horizon(self):
        response = self.client.post('/api/cities/boston/refresh', json={'horizon': 999})
        self.assertEqual(response.status_code, 400)

    def test_forecast_snapshot_is_scoped_to_its_scenario(self):
        def provider(**kwargs):
            return {'city': kwargs['city'], 'datum': 'MSL',
                    'source': {'status': 'historical_fallback'},
                    'history': {'last_observation_utc': '2024-12-31T00:00:00+00:00'},
                    'forecast': [{'hour_ahead': 1, 'p50_m': 0.2}]}
        client = create_app(self.root, live_provider=provider).test_client()
        self.assertEqual(client.post('/api/cities/boston/refresh', json={'scenario': 'plus_50cm'}).status_code, 200)
        data = client.get('/api/cities/boston/analysis?scenario=plus_50cm').get_json()
        self.assertEqual(data['source_status'], 'historical_fallback')
        self.assertEqual(len(data['forecast']), 1)
        self.assertEqual(client.get('/api/cities/boston/analysis?scenario=plus_100cm').get_json()['forecast'], [])


if __name__ == '__main__':
    unittest.main()

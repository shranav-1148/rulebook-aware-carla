"""Dependency-free checks for Milestone 2 map exporter helpers."""
import unittest
from types import SimpleNamespace

from carla_map_export import validate_opendrive, safe_filename, capture_map


class FakeMap:
    name = '/Game/Carla/Maps/Town01'

    def to_opendrive(self):
        return '<OpenDRIVE><header revMajor="1"/><road id="1"/></OpenDRIVE>'

    def get_spawn_points(self):
        return [SimpleNamespace(
            location=SimpleNamespace(x=10.0, y=3.0, z=0.5),
            rotation=SimpleNamespace(yaw=90.0),
        )]


class MapExportTests(unittest.TestCase):
    def test_accepts_opendrive_with_road(self):
        self.assertEqual(validate_opendrive(FakeMap().to_opendrive()), 1)

    def test_rejects_invalid_root_or_missing_roads(self):
        with self.assertRaises(ValueError):
            validate_opendrive('<Other><road/></Other>')
        with self.assertRaises(ValueError):
            validate_opendrive('<OpenDRIVE></OpenDRIVE>')

    def test_filename_no_folders(self):
        self.assertEqual(safe_filename('/Game/Carla/Maps/Town01'), 'Town01')

    def test_export_metadata_and_hash(self):
        text, meta = capture_map(FakeMap(), server_version='fake-test')
        self.assertEqual(meta['opendrive_road_count'], 1)
        self.assertEqual(meta['spawn_points_carla'][0]['y_m'], 3.0)
        self.assertEqual(len(meta['opendrive_sha256']), 64)
        self.assertEqual(meta['carla_server_version'], 'fake-test')


if __name__ == '__main__':
    unittest.main()

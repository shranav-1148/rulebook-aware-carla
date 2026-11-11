"""Export the currently loaded CARLA road map and spawn-point samples.

see PROVENANCE.md.
Read-only with respect to the running CARLA simulation.
Run with the CARLA Python environment on Windows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def validate_opendrive(xml_text: str) -> int:
    """Confirm well-formed OpenDRIVE with at least one road (not full validation)."""
    root = ET.fromstring(xml_text)
    if root.tag.split('}')[-1] != 'OpenDRIVE':
        raise ValueError(f'Expected <OpenDRIVE> root, got {root.tag!r}')
    roads = sum(1 for item in root if item.tag.split('}')[-1] == 'road')
    if roads == 0:
        raise ValueError('OpenDRIVE contains no <road> elements')
    return roads


def safe_filename(map_name: str) -> str:
    """Use a map's basename but no path separators in output filenames."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', map_name.replace('\\', '/').split('/')[-1]) or 'carla_map'


def capture_map(carla_map, *, server_version: str = 'unknown') -> tuple[str, dict]:
    """Read OpenDRIVE and spawn points from a CARLA Map-like object."""
    xodr_text = carla_map.to_opendrive()
    road_count = validate_opendrive(xodr_text)
    samples = []
    for idx, transform in enumerate(carla_map.get_spawn_points()):
        samples.append({
            'index': idx,
            'x_m': float(transform.location.x),
            'y_m': float(transform.location.y),
            'z_m': float(transform.location.z),
            'yaw_deg': float(transform.rotation.yaw),
        })
    metadata = {
        'schema_version': 1,
        'source': 'CARLA map API',
        'carla_map_name': str(carla_map.name),
        'carla_server_version': server_version,
        'opendrive_road_count': road_count,
        'opendrive_sha256': hashlib.sha256(xodr_text.encode('utf-8')).hexdigest(),
        'spawn_points_carla': samples,
        'coordinate_convention': 'CARLA left-handed XY; yaw degrees',
        'note': 'Spawn points are samples to test map alignment, not a complete route or scenario.',
    }
    return xodr_text, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description='Read-only export of the loaded CARLA map')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--expect-map', default='Town01', help='Guard against exporting the wrong loaded map')
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent / 'maps')
    args = parser.parse_args()

    import carla  # Installed in your Windows CARLA Python environment
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    carla_map = client.get_world().get_map()
    short_name = carla_map.name.replace('\\', '/').split('/')[-1]
    if args.expect_map and short_name != args.expect_map:
        raise RuntimeError(f'Loaded map {carla_map.name!r}, expected {args.expect_map!r}. No files written.')

    version = client.get_server_version()
    xodr_text, metadata = capture_map(carla_map, server_version=str(version))
    stem = safe_filename(carla_map.name)
    out_xodr = args.out_dir / f'{stem}.xodr'
    out_meta = args.out_dir / f'{stem}.map_metadata.json'
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if out_xodr.exists() or out_meta.exists():
        raise FileExistsError(f'Refusing to overwrite: {out_xodr} or {out_meta}')

    out_xodr.write_bytes(xodr_text.encode('utf-8'))  # preserve SHA across Windows/WSL
    out_meta.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Map: {carla_map.name} | OpenDRIVE roads: {metadata["opendrive_road_count"]}')
    print(f'Spawn-point samples: {len(metadata["spawn_points_carla"])}')
    print(f'OpenDRIVE: {out_xodr}')
    print(f'Metadata: {out_meta}')
    print('No CARLA actors, world settings, or simulation ticks changed.')


if __name__ == '__main__':
    main()

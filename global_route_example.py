
import sys
import time
from pathlib import Path
from collections import Counter

import carla

# Location of CARLA's Python navigation agents.
# Change this if your installation is elsewhere.
agents_root = Path(r"D:\CARLA_0.9.16\PythonAPI\carla")

if not (agents_root / "agents").is_dir():
    raise RuntimeError(
        f"CARLA navigation agents not found: {agents_root}"
    )

sys.path.insert(0, str(agents_root))

from agents.navigation.global_route_planner import (
    GlobalRoutePlanner
)

# Connect to CARLA.
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(30.0)

world = client.get_world()
carla_map = world.get_map()

if not carla_map.name.endswith("Town01"):
    raise RuntimeError("Load Town01 first.")

# Get possible starting and destination locations.
spawn_points = carla_map.get_spawn_points()

start = spawn_points[0].location

# Build the global route planner.
planner = GlobalRoutePlanner(
    carla_map,
    sampling_resolution=2.0
)

# Search for a destination with a useful route.
route = None
destination = None

for target in spawn_points[1:]:
    try:
        candidate = planner.trace_route(
            start,
            target.location
        )
    except Exception:
        continue

    if 50 <= len(candidate) <= 200:
        route = candidate
        destination = target.location
        break

if route is None:
    raise RuntimeError(
        "Could not find a suitable route in Town01."
    )

print("Route calculated successfully!")
print("Starting location:", start)
print("Destination:", destination)
print("Number of route waypoints:", len(route))

# Count the planned maneuver labels.
maneuvers = Counter(
    option.name for waypoint, option in route
)

print("Route maneuvers:", dict(maneuvers))

# Extract waypoint locations.
locations = [
    waypoint.transform.location
    for waypoint, option in route
]

# Draw the route using blue lines.
for i in range(len(locations) - 1):
    p1 = locations[i]
    p2 = locations[i + 1]

    a = carla.Location(
        x=p1.x, y=p1.y, z=p1.z + 0.5
    )

    b = carla.Location(
        x=p2.x, y=p2.y, z=p2.z + 0.5
    )

    world.debug.draw_line(
        a,
        b,
        thickness=0.15,
        color=carla.Color(0, 100, 255),
        life_time=120.0
    )

# Mark the beginning of the route.
world.debug.draw_string(
    locations[0] + carla.Location(z=1.0),
    "START",
    draw_shadow=False,
    color=carla.Color(0, 255, 0),
    life_time=120.0
)

# Mark the destination.
world.debug.draw_string(
    locations[-1] + carla.Location(z=1.0),
    "DESTINATION",
    draw_shadow=False,
    color=carla.Color(255, 0, 0),
    life_time=120.0
)

# Position the spectator above the route.
xs = [p.x for p in locations]
ys = [p.y for p in locations]

center_x = (min(xs) + max(xs)) / 2
center_y = (min(ys) + max(ys)) / 2

span = max(
    max(xs) - min(xs),
    max(ys) - min(ys)
)

camera_height = max(60.0, span * 1.2)

spectator = world.get_spectator()

spectator.set_transform(
    carla.Transform(
        carla.Location(
            x=center_x,
            y=center_y,
            z=locations[0].z + camera_height
        ),
        carla.Rotation(pitch=-90.0)
    )
)

print("Route drawn in CARLA.")
time.sleep(20)

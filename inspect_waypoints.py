
import carla
import time

# Connect to CARLA.
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

world = client.get_world()
carla_map = world.get_map()

print("Current CARLA map:", carla_map.name)

# Make sure Town01 is loaded.
if not carla_map.name.endswith("Town01"):
    raise RuntimeError("Please load Town01 first.")

# Get all recommended vehicle spawn locations.
spawn_points = carla_map.get_spawn_points()

# Select the first spawn point.
start = spawn_points[0]

# Find the nearest driving-lane waypoint.
waypoint = carla_map.get_waypoint(
    start.location,
    project_to_road=True,
    lane_type=carla.LaneType.Driving
)

if waypoint is None:
    raise RuntimeError("No driving waypoint found.")

print("\nWaypoint information:")
print("Road ID:", waypoint.road_id)
print("Lane ID:", waypoint.lane_id)
print("Lane width:", waypoint.lane_width)
print("Inside junction:", waypoint.is_junction)
print("Location:", waypoint.transform.location)
print("Direction:", waypoint.transform.rotation)

# Move CARLA's spectator camera above the road.
location = waypoint.transform.location

spectator = world.get_spectator()

camera_location = carla.Location(
    x=location.x,
    y=location.y,
    z=location.z + 60.0
)

spectator.set_transform(
    carla.Transform(
        camera_location,
        carla.Rotation(pitch=-90.0)
    )
)

# Draw 50 consecutive waypoints.
current = waypoint
count = 0

for i in range(50):
    point = current.transform.location

    # Draw a green point slightly above the road.
    draw_location = carla.Location(
        x=point.x,
        y=point.y,
        z=point.z + 0.5
    )

    world.debug.draw_point(
        draw_location,
        size=0.2,
        color=carla.Color(0, 255, 0),
        life_time=60.0
    )

    count += 1

    # Ask for waypoints approximately 2 meters ahead.
    choices = current.next(2.0)

    if not choices:
        print("Reached the end of the connected path.")
        break

    if len(choices) > 1:
        print(
            "Multiple possible road continuations:",
            len(choices)
        )

    # For this exercise, follow the first continuation.
    current = choices[0]

print("\nWaypoints drawn:", count)
print("Look at the CARLA simulator window.")

# Keep the script alive briefly for inspection.
time.sleep(10)

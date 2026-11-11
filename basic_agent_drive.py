
import sys
import math
from pathlib import Path

import carla

# Configure CARLA navigation agent imports.
agents_root = Path(r"D:\CARLA_0.9.16\PythonAPI\carla")

if not (agents_root / "agents").is_dir():
    raise RuntimeError(
        f"CARLA navigation agents not found: {agents_root}"
    )

sys.path.insert(0, str(agents_root))

from agents.navigation.basic_agent import BasicAgent

# Connect to CARLA.
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(30.0)

# Reload Town01 to start with a fresh world.
# Do not run Scenic simultaneously.
world = client.load_world("Town01")
world.set_weather(carla.WeatherParameters.ClearNoon)

# This example runs in asynchronous mode.
if world.get_settings().synchronous_mode:
    raise RuntimeError(
        "CARLA is in synchronous mode. "
        "Restart CARLA before running this example."
    )

carla_map = world.get_map()
spawn_points = carla_map.get_spawn_points()

# Choose our Tesla blueprint.
blueprint = world.get_blueprint_library().find(
    "vehicle.tesla.model3"
)

blueprint.set_attribute("role_name", "hero")

vehicle = None

try:
    # Spawn the vehicle.
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(
            blueprint, spawn_point
        )

        if vehicle is not None:
            break

    if vehicle is None:
        raise RuntimeError(
            "No available location to spawn the vehicle."
        )

    print("Vehicle spawned:", vehicle.id)

    # Create CARLA's BasicAgent.
    agent = BasicAgent(
        vehicle,
        target_speed=25
    )

    print("BasicAgent initialized.")

    # Find a destination that requires driving.
    destination = None
    planner = agent.get_global_planner()

    for target in spawn_points:
        if vehicle.get_location().distance(
            target.location
        ) < 40:
            continue

        try:
            route = planner.trace_route(
                vehicle.get_location(),
                target.location
            )
        except Exception:
            continue

        if len(route) >= 50:
            destination = target.location
            break

    if destination is None:
        raise RuntimeError(
            "Could not find a reachable destination."
        )

    agent.set_destination(destination)

    print("Destination:", destination)
    print("Beginning autonomous driving...")

    spectator = world.get_spectator()

    # Main driving loop.
    for step in range(10000):
        if agent.done():
            print("Destination reached!")
            break

        # Ask BasicAgent for its next control action.
        control = agent.run_step()

        # Apply steering, throttle, and brake.
        vehicle.apply_control(control)

        # Find the vehicle's current position.
        transform = vehicle.get_transform()
        position = transform.location
        forward = transform.get_forward_vector()

        # Move the spectator behind the vehicle.
        camera_location = carla.Location(
            x=position.x - forward.x * 8.0,
            y=position.y - forward.y * 8.0,
            z=position.z + 4.0
        )

        camera_rotation = carla.Rotation(
            pitch=-15.0,
            yaw=transform.rotation.yaw
        )

        spectator.set_transform(
            carla.Transform(
                camera_location,
                camera_rotation
            )
        )

        # Print driving information periodically.
        if step % 25 == 0:
            velocity = vehicle.get_velocity()

            speed_mps = math.sqrt(
                velocity.x ** 2 +
                velocity.y ** 2 +
                velocity.z ** 2
            )

            speed_kmh = speed_mps * 3.6
            remaining_distance = vehicle.get_location().distance(
                destination
            )

            remaining_waypoints = len(
                agent.get_local_planner().get_plan()
            )
            print(
                f"Step={step} "
                f"Speed={speed_kmh:.1f} km/h "
                f"Throttle={control.throttle:.2f} "
                f"Brake={control.brake:.2f} "
                f"Steer={control.steer:.2f} "
                f"Distance={remaining_distance:.1f} m "
                f"Waypoints={remaining_waypoints}"
            )

        # Wait for the next server frame.
        world.wait_for_tick(5.0)

    else:
        print("Maximum simulation steps reached.")

finally:
    # Always remove the vehicle when finished.
    if vehicle is not None and vehicle.is_alive:
        vehicle.destroy()
        print("Vehicle cleaned up.")

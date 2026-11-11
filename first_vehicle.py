
import carla
import time

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

# Load a known map.
world = client.load_world("Town01")

# Get the vehicle blueprint.
blueprints = world.get_blueprint_library()
car_bp = blueprints.find("vehicle.tesla.model3")
car_bp.set_attribute("role_name", "hero")

# Obtain valid vehicle spawn positions.
spawn_points = world.get_map().get_spawn_points()

vehicle = None

try:
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(car_bp, spawn_point)
        if vehicle is not None:
            break

    if vehicle is None:
        raise RuntimeError("No available vehicle spawn point")

    print("Vehicle created:", vehicle.id)

    # Let CARLA's Traffic Manager drive the car.
    vehicle.set_autopilot(True)

    # Move spectator camera above the vehicle.
    spectator = world.get_spectator()

    for step in range(100):
        transform = vehicle.get_transform()

        camera_position = transform.location + carla.Location(z=15)
        camera_rotation = carla.Rotation(
            pitch=-75,
            yaw=transform.rotation.yaw
        )

        spectator.set_transform(
            carla.Transform(camera_position, camera_rotation)
        )

        if step % 10 == 0:
            velocity = vehicle.get_velocity()
            speed = (velocity.x**2 + velocity.y**2
                     + velocity.z**2) ** 0.5
            print(f"Speed: {speed:.2f} m/s")

        time.sleep(0.1)

finally:
    if vehicle is not None:
        vehicle.destroy()
        print("Vehicle removed")

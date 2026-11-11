
import carla
import time

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(5.0)

print("Waiting for Scenic to spawn the ego vehicle...")

last_id = None

try:
    while True:
        try:
            # Refresh world reference, as Scenic reloads Town01.
            world = client.get_world()

            vehicles = world.get_actors().filter("vehicle.*")

            ego = next(
                (v for v in vehicles
                 if v.attributes.get("role_name") == "hero"),
                None
            )

            if ego is None:
                if last_id is not None:
                    print("Ego vehicle disappeared.")
                    last_id = None
                time.sleep(0.1)
                continue

            if ego.id != last_id:
                print(f"Found ego vehicle: {ego.id}")
                last_id = ego.id

            # Get the vehicle's current location and heading.
            transform = ego.get_transform()
            forward = transform.get_forward_vector()

            # Position the camera 10 m behind and 6 m above.
            camera_location = carla.Location(
                x=transform.location.x - 10 * forward.x,
                y=transform.location.y - 10 * forward.y,
                z=transform.location.z + 6
            )

            camera_rotation = carla.Rotation(
                pitch=-20,
                yaw=transform.rotation.yaw
            )

            spectator = world.get_spectator()
            spectator.set_transform(
                carla.Transform(
                    camera_location,
                    camera_rotation
                )
            )

        except RuntimeError:
            # A temporary error can occur while Scenic
            # reloads the CARLA world.
            time.sleep(0.2)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nSpectator follower stopped.")

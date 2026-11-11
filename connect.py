
import carla

# Connect to the CARLA server.
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

# Get the active simulation world.
world = client.get_world()

# Inspect the environment.
print("Connected to CARLA!")
print("Map:", world.get_map().name)
print("Available spawn points:", len(
    world.get_map().get_spawn_points()
))

print("Simulation actors:", len(world.get_actors()))


import carla
from pathlib import Path

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(30.0)

world = client.load_world("Town01")

output = Path("maps/Town01.xodr")
output.parent.mkdir(parents=True, exist_ok=True)

world.get_map().save_to_disk(str(output.resolve()))

print("Saved:", output.resolve())

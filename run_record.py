
import csv
import math
from pathlib import Path

import scenic

# Compile our existing Scenic scenario.
scenario = scenic.scenarioFromFile(
    "first_carla.scenic",
    mode2D=True,
    params={"render": 0}
)

# Generate one concrete scene.
scene, iterations = scenario.generate()

print(f"Generated scene in {iterations} iterations")

# Scenic selects the CARLA simulator declared
# inside first_carla.scenic.
simulator = scenario.getSimulator()

try:
    # Run for up to 250 simulation steps.
    simulation = simulator.simulate(
        scene,
        maxSteps=250
    )

    if simulation is None:
        raise RuntimeError("Simulation was rejected")

    records = simulation.result.records

finally:
    # Restore CARLA's simulator settings.
    simulator.destroy()

# The record statements we defined in Scenic.
fields = ("pos_x", "pos_y", "heading_rad", "speed_mps")

# Each series consists of (step, value) pairs.
series = {
    name: dict(records[name])
    for name in fields
}

# Keep the simulation steps present in all series.
steps = sorted(
    set.intersection(
        *(set(series[name]) for name in fields)
    )
)

if not steps:
    raise RuntimeError("No trajectory samples recorded")

dt = float(scene.params.get("timestep", 0.1))

output = Path("results/trajectory.csv")
output.parent.mkdir(parents=True, exist_ok=True)

distance = 0.0
previous_position = None

with output.open("w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "step",
        "time_s",
        "x_m",
        "y_m",
        "heading_rad",
        "speed_mps"
    ])

    for step in steps:
        x = float(series["pos_x"][step])
        y = float(series["pos_y"][step])
        heading = float(series["heading_rad"][step])
        speed = float(series["speed_mps"][step])

        if previous_position is not None:
            px, py = previous_position
            distance += math.hypot(x - px, y - py)

        previous_position = (x, y)

        writer.writerow([
            step,
            round(step * dt, 3),
            x,
            y,
            heading,
            speed
        ])

print("Saved:", output.resolve())
print("Samples:", len(steps))
print(f"Recorded path length: {distance:.2f} m")
print("Termination:", simulation.result.terminationReason)

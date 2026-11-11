
import csv
import matplotlib.pyplot as plt

x_positions = []
y_positions = []

with open("results/trajectory.csv", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        x_positions.append(float(row["x_m"]))
        y_positions.append(float(row["y_m"]))

plt.figure(figsize=(9, 6))

plt.plot(x_positions, y_positions, label="Ego trajectory")
plt.scatter(
    x_positions[0],
    y_positions[0],
    marker="o",
    label="Start"
)
plt.scatter(
    x_positions[-1],
    y_positions[-1],
    marker="x",
    label="End"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("CARLA Ego Vehicle Trajectory")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

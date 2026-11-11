
param map = localPath("maps/Town01.xodr")
param carla_map = "Town01"
param weather = "ClearNoon"

model scenic.simulators.carla.model

lane = Uniform(*network.lanes)

ego = new Car on lane.centerline,
    facing roadDirection,
    with blueprint "vehicle.tesla.model3",
    with rolename "hero",
    with behavior FollowLaneBehavior(8)

# Record the ego vehicle's state at every timestep.
record ego.position.x as pos_x
record ego.position.y as pos_y
record ego.heading as heading_rad
record ego.speed as speed_mps

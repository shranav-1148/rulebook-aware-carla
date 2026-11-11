"""Pure coordinate conversions; no CARLA or CommonRoad dependency.

CARLA: left-handed XY, yaw in degrees.
CommonRoad convention: right-handed XY, yaw in radians.
Only a mirror transform is applied: no map registration/translation.
"""
import math


def xy_to_commonroad(x_m, y_m):
    return [float(x_m), -float(y_m)]


def yaw_to_commonroad(yaw_deg):
    theta = -math.radians(float(yaw_deg))
    return math.atan2(math.sin(theta), math.cos(theta))


def vector_to_commonroad(vx, vy):
    return [float(vx), -float(vy)]


def longitudinal_velocity(vx, vy, yaw_deg):
    """Projection onto the actor's longitudinal forward heading (m/s)."""
    theta = math.radians(float(yaw_deg))
    return float(vx) * math.cos(theta) + float(vy) * math.sin(theta)


def lateral_velocity(vx, vy, yaw_deg):
    theta = math.radians(float(yaw_deg))
    return -float(vx) * math.sin(theta) + float(vy) * math.cos(theta)


def converted_actor(actor):
    pose = actor['pose_carla']
    vel = actor['velocity_carla_mps']
    acc = actor['acceleration_carla_mps2']
    yaw = pose['yaw_deg']
    return {
        'actor_id': int(actor['actor_id']),
        'type_id': actor['type_id'],
        'position_m': xy_to_commonroad(pose['x_m'], pose['y_m']),
        'orientation_rad': yaw_to_commonroad(yaw),
        'velocity_xy_mps': vector_to_commonroad(vel['x'], vel['y']),
        'acceleration_xy_mps2': vector_to_commonroad(acc['x'], acc['y']),
        'longitudinal_velocity_mps': longitudinal_velocity(vel['x'], vel['y'], yaw),
        'lateral_velocity_mps': lateral_velocity(vel['x'], vel['y'], yaw),
        'longitudinal_acceleration_mps2': longitudinal_velocity(acc['x'], acc['y'], yaw),
        'bbox_carla_local': actor['bbox_carla_local'],
    }

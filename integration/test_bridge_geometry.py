import math
import unittest
from bridge_geometry import (
    xy_to_commonroad,
    yaw_to_commonroad,
    vector_to_commonroad,
    longitudinal_velocity,
    lateral_velocity,
    converted_actor,
)


class BridgeGeometryTest(unittest.TestCase):
    def test_mirror_xy(self):
        # Testing mirror conversion from commonroad to carla from coordinates and velocity vector
        self.assertEqual(xy_to_commonroad(4, 2), [4.0, -2.0])
        self.assertEqual(vector_to_commonroad(6, -3), [6.0, 3.0])

    def test_yaw(self):
        # Conversion of yaw from degrees to radians
        self.assertAlmostEqual(yaw_to_commonroad(90), -math.pi / 2)
        self.assertAlmostEqual(yaw_to_commonroad(-90), math.pi / 2)

    def test_forward_velocity(self):
        # Testing longitudinal velocity with different angles
        self.assertAlmostEqual(longitudinal_velocity(0, 8, 90), 8.0)
        self.assertAlmostEqual(longitudinal_velocity(8, 0, 0), 8.0)

    def test_actor_conversion(self):
        # Testing conversion of raw daa into an actor
        record = {
            'actor_id': 5, 'type_id': 'vehicle.test',
            'pose_carla': {'x_m': 12, 'y_m': 3, 'yaw_deg': 90},
            'velocity_carla_mps': {'x': 0, 'y': 6},
            'acceleration_carla_mps2': {'x': 0, 'y': 1},
            'bbox_carla_local': {'half_length_m': 2.0},
        }
        converted = converted_actor(record)
        self.assertEqual(converted['position_m'], [12, -3])
        self.assertAlmostEqual(converted['longitudinal_velocity_mps'], 6)
        self.assertAlmostEqual(converted['orientation_rad'], -math.pi / 2)


    def test_mirror_preserves_vector_length(self):
        # Testing vector length
        vx, vy = 7.0, -3.5
        cr_vx, cr_vy = vector_to_commonroad(vx, vy)

        self.assertAlmostEqual(
            math.hypot(vx, vy),
            math.hypot(cr_vx, cr_vy)
        )

    def test_heading_and_velocity_reflect_consistently(self):
        yaw_deg = 45.0
        speed = 10.0

        angle = math.radians(yaw_deg)

        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        cr_vx, cr_vy = vector_to_commonroad(vx, vy)
        cr_yaw = yaw_to_commonroad(yaw_deg)

        # Testing velocity vector derived from speed and converted to common roads with the original
        # They must be same
        self.assertAlmostEqual(
            cr_vx,
            speed * math.cos(cr_yaw)
        )
        self.assertAlmostEqual(
            cr_vy,
            speed * math.sin(cr_yaw)
        )

    def test_lateral_velocity_is_currently_carla_right_positive(self):
        self.assertAlmostEqual(
            lateral_velocity(8.0, 2.0, 0.0),
            2.0
        )



if __name__ == '__main__':
    unittest.main()

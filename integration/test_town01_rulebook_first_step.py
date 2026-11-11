"""Dependency-free smoke tests for our Milestone 3C integration helpers.

These do NOT execute the CommonRoad research planner.
"""
import math
import unittest
from town01_rulebook_first_step import serialize_costs, state_records, make_empty_safe_r9_class


class FakeState:
    def __init__(self, time_step, x=0.0, y=0.0, acceleration=0.0):
        self.time_step = time_step
        self.position = [x, y]
        self.orientation = math.pi
        self.velocity = 8.0
        self.acceleration = acceleration


class TestOneStepHelpers(unittest.TestCase):
    def test_none_cost_not_replaced_with_zero(self):
        self.assertEqual(serialize_costs({'R1': None, 'R4': 0.0}),
                         {'R1': None, 'R4': 0.0})

    def test_nonfinite_cost_rejected(self):
        with self.assertRaises(ValueError):
            serialize_costs({'R1': float('nan')})

    def test_predicted_states(self):
        rows = state_records([FakeState(0), FakeState(1, x=-0.8)], 0.1)
        self.assertEqual(rows[1]['time_step'], 1)
        self.assertAlmostEqual(rows[1]['time_s'], 0.1)
        self.assertAlmostEqual(rows[1]['x_m'], -0.8)

    def test_single_state_rejected(self):
        with self.assertRaises(RuntimeError):
            state_records([FakeState(0)], 0.1)

    def test_r9_empty_predictions_are_neutral(self):
        from types import SimpleNamespace

        class FakeUpstreamR9:
            def __init__(self, config):
                self.data = None
            def evaluate(self, sample):
                return 7.5

        Guard = make_empty_safe_r9_class(FakeUpstreamR9)
        rule = Guard(None)
        rule.data = SimpleNamespace(predictions=[])
        self.assertEqual(rule.evaluate(object()), 0.0)

    def test_r9_nonempty_predictions_use_upstream_evaluation(self):
        from types import SimpleNamespace

        class FakeUpstreamR9:
            def __init__(self, config):
                self.data = None
            def evaluate(self, sample):
                return 7.5

        Guard = make_empty_safe_r9_class(FakeUpstreamR9)
        rule = Guard(None)
        rule.data = SimpleNamespace(predictions=[object()])
        self.assertEqual(rule.evaluate(object()), 7.5)

    def test_r9_requires_data_before_evaluation(self):
        class FakeUpstreamR9:
            def __init__(self, config):
                self.data = None
            def evaluate(self, sample):
                return 7.5

        Guard = make_empty_safe_r9_class(FakeUpstreamR9)
        with self.assertRaisesRegex(RuntimeError, "not received planning data"):
            Guard(None).evaluate(object())


if __name__ == '__main__':
    unittest.main()

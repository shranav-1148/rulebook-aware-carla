"""Tests for Milestone 4C; no CARLA or CommonRoad runtime is required."""
import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from town01_motion_graph import Settings, synthetic_route
from town01_conflict_sweep import (
    classify_failure, evaluate_case, execute, parse_numeric_values, run_sweep,
)


class TestConflictSweep(unittest.TestCase):
    def setUp(self):
        self.route = synthetic_route()
        self.cfg = Settings()

    def test_parse_values_and_reject_duplicates_nonfinite(self):
        self.assertEqual(parse_numeric_values("12, 21, 30", "stations"), [12.0, 21.0, 30.0])
        for bad in ["12,12", "3,nan", "inf", "", "1,abc"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_numeric_values(bad, "stations")

    def test_successful_conflict_profile_and_same_graph(self):
        rec, detail = evaluate_case(self.route, replace(self.cfg, obstacle_station=21, obstacle_offset=-2.4))
        self.assertEqual(rec["status"], "ok")
        self.assertTrue(rec["same_feasible_graph"])
        self.assertTrue(rec["same_terminal_set"])
        self.assertEqual(rec["states"], 2727)
        self.assertEqual(rec["edges"], 8627)
        self.assertLess(rec["R9_first_R9"], rec["R5_first_R9"])
        self.assertLess(rec["R5_first_R5"], rec["R9_first_R5"])
        self.assertGreater(rec["max_aligned_separation_m"], 10.0)
        self.assertEqual(len(detail["R9_first"]), self.cfg.horizon+1)

    def test_far_away_obstacle_yields_same_path(self):
        rec, _ = evaluate_case(self.route, replace(self.cfg, obstacle_station=65, obstacle_offset=0))
        self.assertEqual(rec["status"], "ok")
        self.assertTrue(rec["same_selected_state_path"])
        self.assertAlmostEqual(rec["max_aligned_separation_m"], 0.0)

    def test_physically_blocked_case_is_not_called_zero_divergence(self):
        rec, detail = evaluate_case(self.route, replace(self.cfg, obstacle_station=21, obstacle_offset=0))
        self.assertIsNone(detail)
        self.assertEqual(rec["status"], "no_full_horizon_path")
        self.assertIsNone(rec["max_aligned_separation_m"])

    def test_terminal_progress_filter_is_shared(self):
        cfg = replace(self.cfg, obstacle_station=21, obstacle_offset=-2.4)
        rec, _ = evaluate_case(self.route, cfg, min_final_s=40.0)
        self.assertEqual(rec["status"], "ok")
        self.assertGreaterEqual(rec["R9_first_final_s_m"], 40.0)
        self.assertGreaterEqual(rec["R5_first_final_s_m"], 40.0)
        self.assertTrue(rec["same_terminal_set"])

    def test_unreachable_terminal_progress_reported(self):
        cfg = replace(self.cfg, obstacle_station=21, obstacle_offset=-2.4)
        rec, detail = evaluate_case(self.route, cfg, min_final_s=70.0)
        self.assertEqual(rec["status"], "no_eligible_terminal")
        self.assertIsNone(detail)

    def test_sweep_grid_and_case_status(self):
        records, details = run_sweep(self.route, self.cfg, [21,65], [-2.4, 0])
        self.assertEqual(len(records), 4)
        self.assertEqual(sum(r["status"] == "ok" for r in records), 3)
        self.assertEqual(len(details), 3)

    def test_only_expected_failures_are_classified(self):
        self.assertEqual(classify_failure(RuntimeError("No feasible states at step 3")), "no_full_horizon_path")
        self.assertIsNone(classify_failure(RuntimeError("unexpected error")))

    def test_export_summary_csv_and_infeasible_counts(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "result"
            summary, records = execute(self.route, "synthetic-demo", self.cfg,
                                       [21,65], [-2.4,0], out)
            self.assertEqual(summary["cases_total"], 4)
            self.assertEqual(summary["cases_infeasible_or_no_terminal"], 1)
            with (out / "scenario_results.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertTrue((out / "sweep_summary.json").exists())
            self.assertTrue((out / "examples" / "baseline_21m_right_2p4m" / "R9_first.csv").exists())


if __name__ == "__main__":
    unittest.main()

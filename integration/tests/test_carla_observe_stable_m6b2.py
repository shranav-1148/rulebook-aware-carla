"""Offline/mock checks for passive CARLA settling capture; no simulator needed."""
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

import carla_observe_stable_m6b2 as s
import test_carla_observe_m6b1 as previous


def mk_pair(frame, time, *, ez=0., ex=0., prop_v=0., x=1., z=1., yaw=0., prop_x=3., prop_z=0.):
    return {'frame': frame, 'time_s': time,
            'ego': {'xyz': (x, 2., z), 'rpy_deg': (0., 0., yaw), 'vxyz': (ex, 0., ez)},
            'obstacle': {'xyz': (prop_x, 5., prop_z), 'rpy_deg': (0., 0., 0.),
                         'vxyz': (prop_v, 0., 0.)}}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.c = s.StabilityCriteria(min_warmup_sim_s=.2, steady_dwell_sim_s=.3, min_steady_samples=3)

    def test_first_frame_never_accepted(self):
        g = s.PassiveStabilityGate(self.c)
        self.assertFalse(g.update(mk_pair(1, 10)))
        self.assertEqual(g.summary()['latest']['steady_samples'], 0)

    def test_stable_after_warmup_dwell_and_min_samples(self):
        g = s.PassiveStabilityGate(self.c)
        events = [(1, 10), (2, 10.25), (3, 10.35), (4, 10.45), (5, 10.55)]
        results = [g.update(mk_pair(*p)) for p in events]
        self.assertEqual(results, [False, False, False, False, True])
        self.assertEqual(g.frames_seen, 5)

    def test_initial_falling_resets_stability(self):
        g = s.PassiveStabilityGate(self.c)
        self.assertFalse(g.update(mk_pair(1, 0, ez=-1)))
        self.assertFalse(g.update(mk_pair(2, .2, ez=-.5)))
        self.assertFalse(g.update(mk_pair(3, .3, ez=0)))
        self.assertFalse(g.update(mk_pair(4, .4, ez=0)))
        self.assertTrue(g.update(mk_pair(5, .6, ez=0)))

    def test_fast_horizontal_velocity_blocks(self):
        g = s.PassiveStabilityGate(self.c)
        for k in range(10):
            self.assertFalse(g.update(mk_pair(k+1, k*.1, ex=.3)))

    def test_obstacle_velocity_blocks(self):
        g = s.PassiveStabilityGate(self.c)
        for k in range(10):
            self.assertFalse(g.update(mk_pair(k+1, k*.1, prop_v=.5)))

    def test_ego_position_drift_resets(self):
        g = s.PassiveStabilityGate(self.c)
        for k, t in enumerate([0,.2,.3]):
            self.assertFalse(g.update(mk_pair(k+1, t)))
        self.assertFalse(g.update(mk_pair(4,.4,x=1.15)))
        self.assertEqual(g.consecutive_stable, 0)

    def test_obstacle_position_drift_resets(self):
        g = s.PassiveStabilityGate(self.c)
        for k,t in enumerate([0,.2,.3]):
            g.update(mk_pair(k+1, t))
        self.assertFalse(g.update(mk_pair(4,.4,prop_x=3.15)))

    def test_yaw_drift_resets(self):
        g = s.PassiveStabilityGate(self.c)
        for k,t in enumerate([0,.2,.3]):
            g.update(mk_pair(k+1, t))
        self.assertFalse(g.update(mk_pair(4,.4,yaw=2.)))

    def test_missing_actor_frame_resets(self):
        g = s.PassiveStabilityGate(self.c)
        for k,t in enumerate([0,.2,.3]):
            g.update(mk_pair(k+1, t))
        g.record_missing(4)
        self.assertEqual(g.summary()['missing_actor_frames'], 1)
        self.assertEqual(g.consecutive_stable, 0)
        self.assertFalse(g.update(mk_pair(5, .5)))

    def test_reject_duplicate_or_reversed_frame(self):
        g = s.PassiveStabilityGate(self.c)
        g.update(mk_pair(12, 1.))
        with self.assertRaisesRegex(RuntimeError, 'strictly increasing'):
            g.update(mk_pair(12, 2.))

    def test_reject_reversed_timestamp(self):
        g = s.PassiveStabilityGate(self.c)
        g.update(mk_pair(12, 1.))
        with self.assertRaisesRegex(RuntimeError, 'timestamps'):
            g.update(mk_pair(13, .9))

    def test_reject_nonfinite_values(self):
        g = s.PassiveStabilityGate(self.c)
        with self.assertRaises(ValueError):
            g.update(mk_pair(1, float('nan')))

    def test_angular_wrap_difference(self):
        self.assertAlmostEqual(s.angle_difference_degrees(179, -179), 2, places=8)

    def test_extract_pair_snapshots_or_none(self):
        ego, prop, snap = previous.Tests.make_pair(previous.Tests('runTest'))
        self.assertIsNotNone(s.extract_pair(snap, ego.id, prop.id))
        self.assertIsNone(s.extract_pair(snap, ego.id, 123456))

    def test_gate_criteria_validation(self):
        with self.assertRaises(ValueError):
            s.PassiveStabilityGate(s.StabilityCriteria(min_steady_samples=1))


class MockCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous.Tests.setUpClass()
        cls.ctx=previous.Tests.context
        cls.carla=previous.Tests.carla

    def setup(self):
        w=previous.FakeWorld()
        bps=(NS(id='vehicle.tesla.model3'),NS(id='static.prop.constructioncone'))
        args=NS(confirm_spawn=True,ego_z_offset=.5,timeout=1.,warmup_s=.2,
                dwell_s=.3,min_samples=3,max_frames=20,max_sim_seconds=3.)
        wp=NS(transform=previous.pose(384.591,-1.993,0))
        info={'world_synchronous':False,'existing_vehicle_count':0,
              'map':'Carla/Maps/Town01', 'carla_client_version':'0.9.16',
              'carla_server_version':'0.9.16'}
        return w,bps,args,wp,info

    def test_success_cleanup_and_observation_preserves_6b2_interface(self):
        w,bps,args,wp,info=self.setup()
        def fake_wait(world, ego, prop, *args, **kwargs):
            return previous.FakeSnapshot([ego,prop]), {'latest':{'accepted':True}}, True
        with patch.object(s, 'wait_for_settled_pair', fake_wait):
            result,obs=s.spawn_and_capture(args,self.ctx,self.carla,w,wp,info,bps)
        self.assertEqual(result['status'],'stable_snapshot_captured')
        self.assertTrue(result['stability_gate_accepted'])
        self.assertEqual(result['post_settle_diagnostics']['failed_checks'], [])
        self.assertTrue(result['cleanup_ok'])
        self.assertEqual(obs['milestone'],'6B-1')
        self.assertEqual(obs['observation_mode'],'temporary_spawn_passive_snapshot')
        self.assertTrue(obs['cleanup_ok'])
        self.assertEqual(w.tick_calls,0)
        self.assertTrue(all(a.destroyed for a in w.actors))
        self.assertFalse(obs['ready_for_live_control'])
        self.assertFalse(obs['obstacle_ground_contact_verified'])

    def test_cone_local_bbox_placement_not_world_relative(self):
        b = NS(location=NS(x=.04, y=-.03, z=.293),
               extent=NS(x=.172, y=.172, z=.293))
        origin, meta = s.cone_origin_from_local_bbox(363.592, -4.405, 2.98, b)
        self.assertAlmostEqual(origin[0], 363.552)
        self.assertAlmostEqual(origin[1], -4.375)
        self.assertAlmostEqual(origin[2], 3.0)
        self.assertEqual(meta['strategy'],
                         'actor_local_bbox_direct_target_no_spawn_world_measurement')
        self.assertFalse(meta['ground_contact_verified'])

    def test_rejects_world_referenced_cone_bbox_center(self):
        b = NS(location=NS(x=363.592, y=-4.405, z=.293),
               extent=NS(x=.172, y=.172, z=.293))
        with self.assertRaisesRegex(RuntimeError, 'plausible actor-local'):
            s.cone_origin_from_local_bbox(363.592, -4.405, 2.98, b)

    def test_cone_placement_avoids_unpublished_spawn_bbox(self):
        # Regression: the old code used get_world_vertices/get_transform
        # immediately after spawn. This simulated API refuses such a read.
        w,bps,args,wp,info=self.setup()
        got_settled_frame = {'value': False}
        actual_bbox_world = s.old.bbox_world
        def bbox_world_after_first_frame(actor, pose):
            if not got_settled_frame['value']:
                raise AssertionError('bbox world geometry read before a published frame')
            return actual_bbox_world(actor, pose)
        def fake_wait(world, ego, prop, *a, **kw):
            got_settled_frame['value'] = True
            return previous.FakeSnapshot([ego,prop]), {'latest':{'accepted':True}}, True
        with patch.object(s.old, 'bbox_world', side_effect=bbox_world_after_first_frame), \
             patch.object(s, 'wait_for_settled_pair', fake_wait):
            result,obs=s.spawn_and_capture(args,self.ctx,self.carla,w,wp,info,bps)
        self.assertEqual(result['status'], 'stable_snapshot_captured')
        self.assertEqual(result['post_settle_diagnostics']['failed_checks'], [])
        self.assertTrue(result['cleanup_ok'])
        self.assertAlmostEqual(obs['obstacle']['center_cr_xy_m'][0], 363.591)
        self.assertAlmostEqual(obs['obstacle']['center_cr_xy_m'][1], 4.393)
        self.assertTrue(all(actor.destroyed for actor in w.actors))
        self.assertEqual(w.tick_calls, 0)

    def test_post_settle_failure_reports_exact_failed_checks(self):
        w,bps,args,wp,info=self.setup()
        def fake_wait(world, ego, prop, *other_args, **kwargs):
            prop.pose.location.x += 1.0
            return previous.FakeSnapshot([ego,prop]), {'latest': {'accepted': True}}, True
        with patch.object(s, 'wait_for_settled_pair', fake_wait):
            result, obs = s.spawn_and_capture(args, self.ctx, self.carla, w, wp, info, bps)
        self.assertIsNone(obs)
        self.assertEqual(result['status'], 'post_settle_validation_failed')
        self.assertTrue(result['stability_gate_accepted'])
        diag = result['post_settle_diagnostics']
        self.assertIn('obstacle_center_matches_offline_location', diag['failed_checks'])
        self.assertFalse(diag['checks']['obstacle_center_matches_offline_location'])
        self.assertGreater(diag['obstacle_center_position_error_m'], 0.9)
        self.assertTrue(result['cleanup_ok'])
        self.assertTrue(all(a.destroyed for a in w.actors))

    def test_stability_failure_has_no_observation_and_still_cleans_up(self):
        w,bps,args,wp,info=self.setup()
        with patch.object(s,'wait_for_settled_pair',return_value=(None,{'latest':{'accepted':False}},False)):
            result,obs=s.spawn_and_capture(args,self.ctx,self.carla,w,wp,info,bps)
        self.assertIsNone(obs)
        self.assertEqual(result['status'],'stability_capture_failed')
        self.assertTrue(result['cleanup_ok'])
        self.assertTrue(all(a.destroyed for a in w.actors))

    def test_spawn_failure_cleans_owned_ego(self):
        w,bps,args,wp,info=self.setup()
        w.can_spawn_prop=False
        result,obs=s.spawn_and_capture(args,self.ctx,self.carla,w,wp,info,bps)
        self.assertIsNone(obs)
        self.assertFalse(result['cleanup_ok'])
        self.assertTrue(w.actors[0].destroyed)

    def test_sync_mode_refused_before_spawning(self):
        w,bps,args,wp,info=self.setup()
        info['world_synchronous']=True
        with self.assertRaises(RuntimeError):
            s.spawn_and_capture(args,self.ctx,self.carla,w,wp,info,bps)
        self.assertEqual(w.actors,[])

    def test_wait_uses_no_explicit_ticks(self):
        w,bps,args,wp,info=self.setup()
        ego=previous.Actor(1,'vehicle.tesla.model3',previous.pose(0,0))
        prop=previous.Actor(2,'static.prop.constructioncone',previous.pose(0,0))
        class StreamingWorld:
            def __init__(self):self.count=0;self.tick_calls=0
            def wait_for_tick(self,timeout):
                self.count+=1
                snap=previous.FakeSnapshot([ego,prop],frame=self.count)
                snap.timestamp.elapsed_seconds=self.count*.1
                return snap
            def tick(self):
                self.tick_calls+=1
                raise AssertionError('no world.tick expected')
        world=StreamingWorld()
        snap,meta,ok=s.wait_for_settled_pair(world,ego,prop,
                    s.StabilityCriteria(min_warmup_sim_s=.2,steady_dwell_sim_s=.3,min_steady_samples=3),
                    max_frames=20,max_elapsed_sim_s=3)
        self.assertTrue(ok)
        self.assertEqual(world.tick_calls,0)
        self.assertGreater(meta['frames_seen'],3)


if __name__=='__main__':
    unittest.main(verbosity=2)

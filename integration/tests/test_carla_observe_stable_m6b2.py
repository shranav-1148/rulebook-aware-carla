"""Offline/mock checks for passive CARLA settling capture; no simulator needed."""
# Import dependency: math supplies types or behavior used by this test module.
import math
# Import dependency: unittest supplies types or behavior used by this test module.
import unittest
# Import dependency: unittest.mock supplies types or behavior used by this test module.
from unittest.mock import patch
# Import dependency: types supplies types or behavior used by this test module.
from types import SimpleNamespace as NS

# Import dependency: carla_observe_stable_m6b2 supplies types or behavior used by this test module.
import carla_observe_stable_m6b2 as s
# Import dependency: test_carla_observe_m6b1 supplies types or behavior used by this test module.
import test_carla_observe_m6b1 as previous


# Helper routine: mk pair; exposes a focused operation or synthetic fixture while keeping test intent readable.
def mk_pair(frame, time, *, ez=0., ex=0., prop_v=0., x=1., z=1., yaw=0., prop_x=3., prop_z=0.):
    return {'frame': frame, 'time_s': time,
            'ego': {'xyz': (x, 2., z), 'rpy_deg': (0., 0., yaw), 'vxyz': (ex, 0., ez)},
            'obstacle': {'xyz': (prop_x, 5., prop_z), 'rpy_deg': (0., 0., 0.),
                         'vxyz': (prop_v, 0., 0.)}}


# Test class GateTests: groups related setup, helpers, and assertions behind one fixture lifecycle.
class GateTests(unittest.TestCase):
    # Fixture lifecycle: create or release the temporary state used only by this test class.
    def setUp(self):
        self.c = s.StabilityCriteria(min_warmup_sim_s=.2, steady_dwell_sim_s=.3, min_steady_samples=3)

    # Test method: verifies first frame never accepted without changing the behavior under test.
    def test_first_frame_never_accepted(self):
        g = s.PassiveStabilityGate(self.c)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(g.update(mk_pair(1, 10)))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(g.summary()['latest']['steady_samples'], 0)

    # Test method: verifies stable after warmup dwell and min samples without changing the behavior under test.
    def test_stable_after_warmup_dwell_and_min_samples(self):
        g = s.PassiveStabilityGate(self.c)
        events = [(1, 10), (2, 10.25), (3, 10.35), (4, 10.45), (5, 10.55)]
        results = [g.update(mk_pair(*p)) for p in events]
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(results, [False, False, False, False, True])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(g.frames_seen, 5)

    # Test method: verifies initial falling resets stability without changing the behavior under test.
    def test_initial_falling_resets_stability(self):
        g = s.PassiveStabilityGate(self.c)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(g.update(mk_pair(1, 0, ez=-1)))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(g.update(mk_pair(2, .2, ez=-.5)))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(g.update(mk_pair(3, .3, ez=0)))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(g.update(mk_pair(4, .4, ez=0)))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(g.update(mk_pair(5, .6, ez=0)))

    # Test method: verifies fast horizontal velocity blocks without changing the behavior under test.
    def test_fast_horizontal_velocity_blocks(self):
        g = s.PassiveStabilityGate(self.c)
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for k in range(10):
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertFalse(g.update(mk_pair(k+1, k*.1, ex=.3)))

    # Test method: verifies obstacle velocity blocks without changing the behavior under test.
    def test_obstacle_velocity_blocks(self):
        g = s.PassiveStabilityGate(self.c)
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for k in range(10):
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertFalse(g.update(mk_pair(k+1, k*.1, prop_v=.5)))

    # Test method: verifies ego position drift resets without changing the behavior under test.
    def test_ego_position_drift_resets(self):
        g = s.PassiveStabilityGate(self.c)
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for k, t in enumerate([0,.2,.3]):
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertFalse(g.update(mk_pair(k+1, t)))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(g.update(mk_pair(4,.4,x=1.15)))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(g.consecutive_stable, 0)

    # Test method: verifies obstacle position drift resets without changing the behavior under test.
    def test_obstacle_position_drift_resets(self):
        g = s.PassiveStabilityGate(self.c)
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for k,t in enumerate([0,.2,.3]):
            g.update(mk_pair(k+1, t))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(g.update(mk_pair(4,.4,prop_x=3.15)))

    # Test method: verifies yaw drift resets without changing the behavior under test.
    def test_yaw_drift_resets(self):
        g = s.PassiveStabilityGate(self.c)
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for k,t in enumerate([0,.2,.3]):
            g.update(mk_pair(k+1, t))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(g.update(mk_pair(4,.4,yaw=2.)))

    # Test method: verifies missing actor frame resets without changing the behavior under test.
    def test_missing_actor_frame_resets(self):
        g = s.PassiveStabilityGate(self.c)
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for k,t in enumerate([0,.2,.3]):
            g.update(mk_pair(k+1, t))
        g.record_missing(4)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(g.summary()['missing_actor_frames'], 1)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(g.consecutive_stable, 0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(g.update(mk_pair(5, .5)))

    # Test method: verifies reject duplicate or reversed frame without changing the behavior under test.
    def test_reject_duplicate_or_reversed_frame(self):
        g = s.PassiveStabilityGate(self.c)
        g.update(mk_pair(12, 1.))
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(RuntimeError, 'strictly increasing'):
            g.update(mk_pair(12, 2.))

    # Test method: verifies reject reversed timestamp without changing the behavior under test.
    def test_reject_reversed_timestamp(self):
        g = s.PassiveStabilityGate(self.c)
        g.update(mk_pair(12, 1.))
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(RuntimeError, 'timestamps'):
            g.update(mk_pair(13, .9))

    # Test method: verifies reject nonfinite values without changing the behavior under test.
    def test_reject_nonfinite_values(self):
        g = s.PassiveStabilityGate(self.c)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):
            g.update(mk_pair(1, float('nan')))

    # Test method: verifies angular wrap difference without changing the behavior under test.
    def test_angular_wrap_difference(self):
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(s.angle_difference_degrees(179, -179), 2, places=8)

    # Test method: verifies extract pair snapshots or none without changing the behavior under test.
    def test_extract_pair_snapshots_or_none(self):
        ego, prop, snap = previous.Tests.make_pair(previous.Tests('runTest'))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIsNotNone(s.extract_pair(snap, ego.id, prop.id))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIsNone(s.extract_pair(snap, ego.id, 123456))

    # Test method: verifies gate criteria validation without changing the behavior under test.
    def test_gate_criteria_validation(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):
            s.PassiveStabilityGate(s.StabilityCriteria(min_steady_samples=1))


# Test class MockCaptureTests: groups related setup, helpers, and assertions behind one fixture lifecycle.
class MockCaptureTests(unittest.TestCase):
    @classmethod
    # Helper routine: setUpClass; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def setUpClass(cls):
        previous.Tests.setUpClass()
        cls.ctx=previous.Tests.context
        cls.carla=previous.Tests.carla

    # Fixture lifecycle: create or release the temporary state used only by this test class.
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

    # Test method: verifies success cleanup and observation preserves 6b2 interface without changing the behavior under test.
    def test_success_cleanup_and_observation_preserves_6b2_interface(self):
        w,bps,args,wp,info=self.setup()
        # Helper routine: fake wait; exposes a focused operation or synthetic fixture while keeping test intent readable.
        def fake_wait(world, ego, prop, *args, **kwargs):
            return previous.FakeSnapshot([ego,prop]), {'latest':{'accepted':True}}, True
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch.object(s, 'wait_for_settled_pair', fake_wait):
            result,obs=s.spawn_and_capture(args,self.ctx,self.carla,w,wp,info,bps)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(result['status'],'stable_snapshot_captured')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(result['stability_gate_accepted'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(result['post_settle_diagnostics']['failed_checks'], [])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(result['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(obs['milestone'],'6B-1')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(obs['observation_mode'],'temporary_spawn_passive_snapshot')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(obs['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(w.tick_calls,0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(all(a.destroyed for a in w.actors))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(obs['ready_for_live_control'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(obs['obstacle_ground_contact_verified'])

    # Test method: verifies cone local bbox placement not world relative without changing the behavior under test.
    def test_cone_local_bbox_placement_not_world_relative(self):
        b = NS(location=NS(x=.04, y=-.03, z=.293),
               extent=NS(x=.172, y=.172, z=.293))
        origin, meta = s.cone_origin_from_local_bbox(363.592, -4.405, 2.98, b)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(origin[0], 363.552)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(origin[1], -4.375)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(origin[2], 3.0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(meta['strategy'],
                         'actor_local_bbox_direct_target_no_spawn_world_measurement')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(meta['ground_contact_verified'])

    # Test method: verifies rejects world referenced cone bbox center without changing the behavior under test.
    def test_rejects_world_referenced_cone_bbox_center(self):
        b = NS(location=NS(x=363.592, y=-4.405, z=.293),
               extent=NS(x=.172, y=.172, z=.293))
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(RuntimeError, 'plausible actor-local'):
            s.cone_origin_from_local_bbox(363.592, -4.405, 2.98, b)

    # Test method: verifies cone placement avoids unpublished spawn bbox without changing the behavior under test.
    def test_cone_placement_avoids_unpublished_spawn_bbox(self):
        # Regression: the old code used get_world_vertices/get_transform
        # immediately after spawn. This simulated API refuses such a read.
        w,bps,args,wp,info=self.setup()
        got_settled_frame = {'value': False}
        actual_bbox_world = s.old.bbox_world
        # Helper routine: bbox world after first frame; exposes a focused operation or synthetic fixture while keeping test intent readable.
        def bbox_world_after_first_frame(actor, pose):
            # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
            if not got_settled_frame['value']:
                raise AssertionError('bbox world geometry read before a published frame')
            return actual_bbox_world(actor, pose)
        # Helper routine: fake wait; exposes a focused operation or synthetic fixture while keeping test intent readable.
        def fake_wait(world, ego, prop, *a, **kw):
            got_settled_frame['value'] = True
            return previous.FakeSnapshot([ego,prop]), {'latest':{'accepted':True}}, True
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch.object(s.old, 'bbox_world', side_effect=bbox_world_after_first_frame), \
             patch.object(s, 'wait_for_settled_pair', fake_wait):
            result,obs=s.spawn_and_capture(args,self.ctx,self.carla,w,wp,info,bps)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(result['status'], 'stable_snapshot_captured')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(result['post_settle_diagnostics']['failed_checks'], [])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(result['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(obs['obstacle']['center_cr_xy_m'][0], 363.591)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(obs['obstacle']['center_cr_xy_m'][1], 4.393)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(all(actor.destroyed for actor in w.actors))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(w.tick_calls, 0)

    # Test method: verifies post settle failure reports exact failed checks without changing the behavior under test.
    def test_post_settle_failure_reports_exact_failed_checks(self):
        w,bps,args,wp,info=self.setup()
        # Helper routine: fake wait; exposes a focused operation or synthetic fixture while keeping test intent readable.
        def fake_wait(world, ego, prop, *other_args, **kwargs):
            prop.pose.location.x += 1.0
            return previous.FakeSnapshot([ego,prop]), {'latest': {'accepted': True}}, True
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch.object(s, 'wait_for_settled_pair', fake_wait):
            result, obs = s.spawn_and_capture(args, self.ctx, self.carla, w, wp, info, bps)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIsNone(obs)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(result['status'], 'post_settle_validation_failed')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(result['stability_gate_accepted'])
        diag = result['post_settle_diagnostics']
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('obstacle_center_matches_offline_location', diag['failed_checks'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(diag['checks']['obstacle_center_matches_offline_location'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertGreater(diag['obstacle_center_position_error_m'], 0.9)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(result['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(all(a.destroyed for a in w.actors))

    # Test method: verifies stability failure has no observation and still cleans up without changing the behavior under test.
    def test_stability_failure_has_no_observation_and_still_cleans_up(self):
        w,bps,args,wp,info=self.setup()
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch.object(s,'wait_for_settled_pair',return_value=(None,{'latest':{'accepted':False}},False)):
            result,obs=s.spawn_and_capture(args,self.ctx,self.carla,w,wp,info,bps)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIsNone(obs)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(result['status'],'stability_capture_failed')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(result['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(all(a.destroyed for a in w.actors))

    # Test method: verifies spawn failure cleans owned ego without changing the behavior under test.
    def test_spawn_failure_cleans_owned_ego(self):
        w,bps,args,wp,info=self.setup()
        w.can_spawn_prop=False
        result,obs=s.spawn_and_capture(args,self.ctx,self.carla,w,wp,info,bps)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIsNone(obs)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(result['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(w.actors[0].destroyed)

    # Test method: verifies sync mode refused before spawning without changing the behavior under test.
    def test_sync_mode_refused_before_spawning(self):
        w,bps,args,wp,info=self.setup()
        info['world_synchronous']=True
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(RuntimeError):
            s.spawn_and_capture(args,self.ctx,self.carla,w,wp,info,bps)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(w.actors,[])

    # Test method: verifies wait uses no explicit ticks without changing the behavior under test.
    def test_wait_uses_no_explicit_ticks(self):
        w,bps,args,wp,info=self.setup()
        ego=previous.Actor(1,'vehicle.tesla.model3',previous.pose(0,0))
        prop=previous.Actor(2,'static.prop.constructioncone',previous.pose(0,0))
        # Test class StreamingWorld: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class StreamingWorld:
            # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def __init__(self):self.count=0;self.tick_calls=0
            # Helper routine: wait for tick; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def wait_for_tick(self,timeout):
                self.count+=1
                snap=previous.FakeSnapshot([ego,prop],frame=self.count)
                snap.timestamp.elapsed_seconds=self.count*.1
                return snap
            # Helper routine: tick; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def tick(self):
                self.tick_calls+=1
                raise AssertionError('no world.tick expected')
        world=StreamingWorld()
        snap,meta,ok=s.wait_for_settled_pair(world,ego,prop,
                    s.StabilityCriteria(min_warmup_sim_s=.2,steady_dwell_sim_s=.3,min_steady_samples=3),
                    max_frames=20,max_elapsed_sim_s=3)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(ok)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(world.tick_calls,0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertGreater(meta['frames_seen'],3)


# Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
if __name__=='__main__':
    unittest.main(verbosity=2)

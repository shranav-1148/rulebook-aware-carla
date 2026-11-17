import unittest
from carla_compare_paired import calculate, stats, interp

def row(t,px=0,py=0,ax=0,ay=0,yaw=0):
    return {'t_s':float(t),'ref_x_cr_m':float(px),'ref_y_cr_m':float(py),
            'actual_x_cr_m':float(ax),'actual_y_cr_m':float(ay),'ref_yaw_cr_deg':float(yaw)}

class PairedTests(unittest.TestCase):
    def test_equal_planned_and_executed(self):
        a=[row(0,0,0,0,0),row(1,1,1,1,1)]
        b=[row(0,0,0,0,0),row(1,1,0,1,0)]
        data=calculate(a,b)
        self.assertAlmostEqual(stats(data)['max_planned_separation_m'],1)
        self.assertAlmostEqual(stats(data)['max_executed_separation_m'],1)
        self.assertAlmostEqual(stats(data)['divergence_vector_error_rmse_m'],0)
    def test_common_mode_error_cancels(self):
        a=[row(0,0,0,0,0),row(1,1,1,2,1)]
        b=[row(0,0,0,0,0),row(1,1,0,2,0)]
        self.assertAlmostEqual(stats(calculate(a,b))['max_divergence_vector_error_m'],0)
    def test_distinct_errors_change_execution(self):
        a=[row(0),row(1,1,1,1,2)]
        b=[row(0),row(1,1,0,1,0)]
        result=stats(calculate(a,b))
        self.assertAlmostEqual(result['max_planned_separation_m'],1)
        self.assertAlmostEqual(result['max_executed_separation_m'],2)
    def test_interpolation_mismatch(self):
        a=[row(0),row(0.5,0.5,0,0.5,0),row(1,1,0,1,0)]
        b=[row(0),row(1,1,0,1,0)]
        result=calculate(a,b)
        self.assertEqual(len(result),3)
        self.assertAlmostEqual(result[1]['executed_separation_m'],0)
    def test_no_overlap(self):
        with self.assertRaises(ValueError):
            calculate([row(0),row(1)],[row(2),row(3)])
    def test_yaw_wrap(self):
        r=interp([row(0,yaw=179),row(1,yaw=-179)], [0,1],.5)
        self.assertAlmostEqual(r['ref_yaw_cr_deg'],180)

if __name__=='__main__': unittest.main()

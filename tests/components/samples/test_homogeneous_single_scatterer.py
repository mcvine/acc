#!/usr/bin/env python

import os, pytest, numpy as np
thisdir = os.path.abspath(os.path.dirname(__file__))
from mcvine.acc import test

script = os.path.join(thisdir, 'acc_ss_instrument.py')
workdir = 'out.acc_ss'
ncount = int(1e6)

from mcvine.acc import run_script

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_total_time_in_shape():
    from mcvine.acc.components.samples.homogeneous_single_scatterer import total_time_in_shape, time_to_enter, calc_time_to_point_of_scattering
    ts = np.array([-1, 1, 2, 3])
    assert total_time_in_shape(ts, 4)==2
    assert time_to_enter(ts, 4) == 0
    assert total_time_in_shape(ts, 2)==1
    assert time_to_enter(ts, 2) == 0
    assert calc_time_to_point_of_scattering(ts, 4, 0.49) == 0.49*2
    assert calc_time_to_point_of_scattering(ts, 4, 0.51) == 0.51*2 + 1.0
    assert calc_time_to_point_of_scattering(ts, 4, 0.25) == 0.5
    assert calc_time_to_point_of_scattering(ts, 4, 0.75) == 2.5
    ts = np.array([1, 2])
    assert total_time_in_shape(ts, 2)==1
    assert time_to_enter(ts, 2) == 1
    ts = np.array([1, 2, 3, 3.5, 10, 12])
    assert total_time_in_shape(ts, 6)==3.5
    assert time_to_enter(ts, 6) == 1
    ts = np.array([-10, -9, 1, 2, 3, 4])
    assert total_time_in_shape(ts, 6)==2
    assert time_to_enter(ts, 6) == 1
    ts = np.array([-10, -9, -4, -3])
    assert total_time_in_shape(ts, 4)==0
    assert time_to_enter(ts, 4) == 0
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compile():
    run_script.compile(script)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_run():
    run_script.run(script, workdir, ncount=ncount)
    return


def main():
    test_compile()
    test_run()
    return

if __name__ == '__main__': main()

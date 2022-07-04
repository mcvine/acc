#!/usr/bin/env python

import pytest, os, math
from mcvine.acc import test

import os, numpy as np, time
from numba import cuda
from mcvine.acc.geometry import locate, location, arrow_intersect

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_cu_device_locate_wrt_sphere():
    assert locate.cu_device_locate_wrt_sphere(0,0,0, 0.02)==location.inside
    assert locate.cu_device_locate_wrt_sphere(0,0,0.01, 0.02)==location.inside
    assert locate.cu_device_locate_wrt_sphere(0,0,0.02, 0.02)==location.onborder
    assert locate.cu_device_locate_wrt_sphere(0,0,0.03, 0.02)==location.outside
    assert locate.cu_device_locate_wrt_sphere(0,0.01,0, 0.02)==location.inside
    assert locate.cu_device_locate_wrt_sphere(0,0.02,0, 0.02)==location.onborder
    assert locate.cu_device_locate_wrt_sphere(0,0.03,0, 0.02)==location.outside
    assert locate.cu_device_locate_wrt_sphere(0.01,0,0, 0.02)==location.inside
    assert locate.cu_device_locate_wrt_sphere(0.02,0,0, 0.02)==location.onborder
    assert locate.cu_device_locate_wrt_sphere(0.03,0,0, 0.02)==location.outside
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_cu_device_intersect_sphere():
    np.testing.assert_array_equal(
        arrow_intersect.cu_device_intersect_sphere(0,0,0, 0,0,1, 1.),
        [-1, 1]
    )
    np.testing.assert_array_equal(
        arrow_intersect.cu_device_intersect_sphere(0.5,0,0, 0,0,1, 1.),
        [-math.sqrt(3)/2, math.sqrt(3)/2]
    )
    r = arrow_intersect.cu_device_intersect_sphere(1.5,0,0, 0,0,1, 1.)
    assert np.isnan(r[0]) and np.isnan(r[1])
    return

def main():
    test_cu_device_locate_wrt_sphere()
    test_cu_device_intersect_sphere()
    return

if __name__ == '__main__': main()

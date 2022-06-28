#!/usr/bin/env python

import pytest, os
from mcvine.acc import test

import os, math, numpy as np, time
from numba import cuda
from mcvine.acc.geometry import locate, location, arrow_intersect

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_cu_device_locate_wrt_cylinder():
    assert locate.cu_device_locate_wrt_cylinder(0,0,0, 0.02, 0.04)==location.inside
    assert locate.cu_device_locate_wrt_cylinder(0.01,0,0, 0.02, 0.04)==location.inside
    assert locate.cu_device_locate_wrt_cylinder(0.02,0,0, 0.02, 0.04)==location.onborder
    assert locate.cu_device_locate_wrt_cylinder(0.03,0,0, 0.02, 0.04)==location.outside
    assert locate.cu_device_locate_wrt_cylinder(0,0.01,0, 0.02, 0.04)==location.inside
    assert locate.cu_device_locate_wrt_cylinder(0,0.019999,0, 0.02, 0.04)==location.inside
    assert locate.cu_device_locate_wrt_cylinder(0,0.02,0, 0.02, 0.04)==location.onborder
    assert locate.cu_device_locate_wrt_cylinder(0,0.03,0, 0.02, 0.04)==location.outside
    assert locate.cu_device_locate_wrt_cylinder(0,0.02001,0, 0.02, 0.04)==location.outside
    assert locate.cu_device_locate_wrt_cylinder(0,0,0.02, 0.02, 0.04)==location.onborder
    assert locate.cu_device_locate_wrt_cylinder(0,0,0.019999, 0.02, 0.04)==location.inside
    assert locate.cu_device_locate_wrt_cylinder(0,0,0.020001, 0.02, 0.04)==location.outside
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_cu_device_intersect_cylinder():
    np.testing.assert_array_equal(
        arrow_intersect.cu_device_intersect_cylinder(0,0,-5, 0,0,1, 1., 2.),
        [4, 6]
    )
    np.testing.assert_array_equal(
        arrow_intersect.cu_device_intersect_cylinder(0,0,-5, 0,0,1, 1., 1.),
        [4.5, 5.5]
    )
    np.testing.assert_array_equal(
        arrow_intersect.cu_device_intersect_cylinder(0.22,0.35,-5, 0,0,1, 1., 1.),
        [4.5, 5.5]
    )
    x,y,z = math.sqrt(2)/2,-5,0.17
    t1 = 5-math.sqrt(2)/2
    t2 = 5+math.sqrt(2)/2
    np.testing.assert_array_equal(
        arrow_intersect.cu_device_intersect_cylinder(x,y,z, 0,1,0, 1., 1.),
        [t1, t2]
    )
    return

def main():
    test_cu_device_locate_wrt_cylinder()
    test_cu_device_intersect_cylinder()
    return

if __name__ == '__main__': main()

#!/usr/bin/env python

import pytest, os
from mcvine.acc import test

import os, numpy as np, time
from numba import cuda
from mcvine.acc.geometry import oncylinder, location

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_cu_device_locate_wrt_cylinder():
    assert oncylinder.cu_device_locate_wrt_cylinder(0,0,0, 0.02, 0.04)==location.inside
    assert oncylinder.cu_device_locate_wrt_cylinder(0.01,0,0, 0.02, 0.04)==location.inside
    assert oncylinder.cu_device_locate_wrt_cylinder(0.02,0,0, 0.02, 0.04)==location.onborder
    assert oncylinder.cu_device_locate_wrt_cylinder(0.03,0,0, 0.02, 0.04)==location.outside
    assert oncylinder.cu_device_locate_wrt_cylinder(0,0.01,0, 0.02, 0.04)==location.inside
    assert oncylinder.cu_device_locate_wrt_cylinder(0,0.019999,0, 0.02, 0.04)==location.inside
    assert oncylinder.cu_device_locate_wrt_cylinder(0,0.02,0, 0.02, 0.04)==location.onborder
    assert oncylinder.cu_device_locate_wrt_cylinder(0,0.03,0, 0.02, 0.04)==location.outside
    assert oncylinder.cu_device_locate_wrt_cylinder(0,0.02001,0, 0.02, 0.04)==location.outside
    assert oncylinder.cu_device_locate_wrt_cylinder(0,0,0.02, 0.02, 0.04)==location.onborder
    assert oncylinder.cu_device_locate_wrt_cylinder(0,0,0.019999, 0.02, 0.04)==location.inside
    assert oncylinder.cu_device_locate_wrt_cylinder(0,0,0.020001, 0.02, 0.04)==location.outside
    return

def main():
    test_cu_device_locate_wrt_cylinder()
    return

if __name__ == '__main__': main()

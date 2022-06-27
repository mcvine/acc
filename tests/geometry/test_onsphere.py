#!/usr/bin/env python

import pytest, os
from mcvine.acc import test

import os, numpy as np, time
from numba import cuda
from mcvine.acc.geometry import onsphere

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_cu_device_inside_sphere():
    assert onsphere.cu_device_inside_sphere(0,0,0, 0.02)
    assert onsphere.cu_device_inside_sphere(0,0,0.01, 0.02)
    assert not onsphere.cu_device_inside_sphere(0,0,0.02, 0.02)
    assert not onsphere.cu_device_inside_sphere(0,0,0.03, 0.02)
    assert onsphere.cu_device_inside_sphere(0,0.01,0, 0.02)
    assert not onsphere.cu_device_inside_sphere(0,0.02,0, 0.02)
    assert not onsphere.cu_device_inside_sphere(0,0.03,0, 0.02)
    assert onsphere.cu_device_inside_sphere(0.01,0,0, 0.02)
    assert not onsphere.cu_device_inside_sphere(0.02,0,0, 0.02)
    assert not onsphere.cu_device_inside_sphere(0.03,0,0, 0.02)
    return

def main():
    test_cu_device_inside_sphere()
    return

if __name__ == '__main__': main()

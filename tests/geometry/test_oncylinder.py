#!/usr/bin/env python

import pytest, os
from mcvine.acc import test

import os, numpy as np, time
from numba import cuda
from mcvine.acc.geometry import oncylinder, location

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_cu_device_locate_wrt_cylinder():
    assert oncylinder.cu_device_locate_wrt_cylinder(0,0,0, 0.02, 0.01)==location.inside
    return

def main():
    test_cu_device_locate_wrt_cylinder()
    return

if __name__ == '__main__': main()

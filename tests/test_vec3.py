#!/usr/bin/env python

import pytest, os
from mcvine.acc import test

import os, math, numpy as np, time
from numba import cuda
from mcvine.acc import vec3

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_rotate():
    v = np.array([1., 0, 0])
    c = np.array([0, 1., 0])
    angle = math.pi/2
    vec3.rotate(v, c, angle)
    assert np.allclose(v, [0,0,-1])
    vec3.rotate(v, np.array([0,0,1]), math.pi/3)
    assert np.allclose(v, [0,0,-1])
    vec3.rotate(v, np.array([1,0,0]), math.pi)
    assert np.allclose(v, [0,0,1])
    vec3.rotate(v, np.array([1,0,0]), math.pi/3)
    assert np.allclose(v, [0,-math.sqrt(3)/2,1./2])
    return

def main():
    test_rotate()
    return

if __name__ == '__main__': main()

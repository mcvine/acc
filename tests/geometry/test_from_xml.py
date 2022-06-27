#!/usr/bin/env python

import pytest, os

import os, numpy as np, time
from numba import cuda
from math import ceil

from mcvine.acc import test
from mcvine.acc.geometry import locate, location

thisdir = os.path.dirname(__file__)

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_union_example1():
    from instrument.nixml import parse_file
    parsed = parse_file(os.path.join(thisdir, 'union_example1.xml'))
    union1 = parsed[0]
    f = locate.LocateFuncFactory()
    cudadevfunc = f.render(union1)
    assert cudadevfunc(0,0,0) == location.inside
    assert cudadevfunc(0,0,0.03) == location.inside
    assert cudadevfunc(0,0,0.049999) == location.inside
    assert cudadevfunc(0,0,0.05) == location.onborder
    assert cudadevfunc(0,0,0.050001) == location.outside
    assert cudadevfunc(0,0.0249999, 0) == location.inside
    assert cudadevfunc(0,0.025, 0) == location.onborder
    assert cudadevfunc(0,0.0250001, 0) == location.outside
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_union_example1_kernel():
    from instrument.nixml import parse_file
    parsed = parse_file(os.path.join(thisdir, 'union_example1.xml'))
    union1 = parsed[0]
    f = locate.LocateFuncFactory()
    cudadevfunc = f.render(union1)
    @cuda.jit
    def kernel(points, locations):
        idx = cuda.grid(1)
        if idx < len(points):
            x,y,z = points[idx]
            locations[idx] = cudadevfunc(x,y,z)
    points = np.array([
        (0,0,0) ,
        (0,0,0.03) ,
        (0,0,0.049999) ,
        (0,0,0.05) ,
        (0,0,0.050001) ,
        (0,0.0249999, 0) ,
        (0,0.025, 0) ,
        (0,0.0250001, 0) ,
    ])
    expected = np.array([
        location.inside,
        location.inside,
        location.inside,
        location.onborder,
        location.outside,
        location.inside,
        location.onborder,
        location.outside,
    ])
    N = len(points)
    locations = np.zeros(N, dtype=int)
    threadsperblock = 2
    nblocks = ceil(N/threadsperblock)
    kernel[nblocks, threadsperblock](points, locations)
    np.testing.assert_array_equal(locations, expected)
    return

def main():
    test_union_example1()
    test_union_example1_kernel()
    return

if __name__ == '__main__': main()

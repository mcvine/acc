#!/usr/bin/env python

import pytest, os

import os, numpy as np, time
from numba import cuda
from math import ceil

from mcvine.acc import test
from mcvine.acc.geometry import locate, location, arrow_intersect

thisdir = os.path.dirname(__file__)

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_union_example1():
    from instrument.nixml import parse_file
    parsed = parse_file(os.path.join(thisdir, 'union_example1.xml'))
    union1 = parsed[0]
    f = locate.LocateFuncFactory()
    devf_locate = f.render(union1)
    assert devf_locate(0,0,0) == location.inside
    assert devf_locate(0,0,0.03) == location.inside
    assert devf_locate(0,0,0.049999) == location.inside
    assert devf_locate(0,0,0.05) == location.onborder
    assert devf_locate(0,0,0.050001) == location.outside
    assert devf_locate(0,0.0249999, 0) == location.inside
    assert devf_locate(0,0.025, 0) == location.onborder
    assert devf_locate(0,0.0250001, 0) == location.outside
    f = arrow_intersect.ArrowIntersectFuncFactory()
    devf_arrow_intersect = f.render(union1)
    ts = np.zeros(10)
    devf_arrow_intersect(0,0,0, 0,0,1., ts, 0)
    print(ts)
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_union_example1_kernel():
    from instrument.nixml import parse_file
    parsed = parse_file(os.path.join(thisdir, 'union_example1.xml'))
    union1 = parsed[0]
    f = locate.LocateFuncFactory()
    devf_locate = f.render(union1)
    @cuda.jit
    def locate_kernel(points, locations):
        idx = cuda.grid(1)
        if idx < len(points):
            x,y,z = points[idx]
            locations[idx] = devf_locate(x,y,z)
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
    locate_kernel[nblocks, threadsperblock](points, locations)
    np.testing.assert_array_equal(locations, expected)
    # intersect
    f = arrow_intersect.ArrowIntersectFuncFactory()
    devf_intersect = f.render(union1)
    @cuda.jit
    def intersect_kernel(points, velocities, intersections, nintersections):
        idx = cuda.grid(1)
        if idx < len(points):
            x,y,z = points[idx]
            vx,vy,vz = velocities[idx]
            n = devf_intersect(x,y,z, vx,vy,vz, intersections[idx], 0)
            nintersections[idx] = n
    points = np.array([
        (0.,0.,0.),
        (0.,0.,0.),
    ])
    velocities = np.array([
        (0.,0.,1.),
        (1.,0.,0.),
    ])
    expected = np.array([
        (-0.05, 0.05),
        (-0.025, 0.025),
    ])
    npts = len(points)
    intersections = np.zeros((npts, 10), dtype=float)
    nintersections = np.zeros(npts, dtype=int)
    intersect_kernel[nblocks, threadsperblock](
        points, velocities, intersections, nintersections)
    # for i in range(npts):
    #     print(intersections[i, :nintersections[i]])
    np.testing.assert_array_equal(intersections[:, :2], expected)
    return

def main():
    test_union_example1()
    test_union_example1_kernel()
    return

if __name__ == '__main__': main()

#!/usr/bin/env python

import pytest, os

import os, numpy as np, math, time
from numba import cuda
from math import ceil
from instrument.nixml import parse_file

from mcvine.acc import test
from mcvine.acc.geometry import locate, location, arrow_intersect

thisdir = os.path.dirname(__file__)
threadsperblock = 2

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_union_example1():
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
    N = devf_arrow_intersect(0,0,0, 0,0,1., ts, 0)
    # print(ts[:N])
    np.testing.assert_allclose(ts[:N], [-0.05, 0.05])
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_union_example1_kernel():
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
        (0.001,0.001,0.),
        (0.005,0.005,0.),
        (-0.005,0.005,0.),
        (0.,0.,-5.),
        (0.,0.,0.),
        (0.,0.,0.),
        (0.,0.,0.),
    ])
    velocities = np.array([
        (0.,0.,1.),
        (0.,0.,1.),
        (0.,0.,1.),
        (0.,0.,1.),
        (0.,0.,1.),
        (1.,0.,0.),
        (0.,1.,0.),
        (math.sqrt(2),math.sqrt(2),0.),
    ])
    expected = np.array([
        (-0.05, 0.05),
        (-0.05, 0.05),
        (-0.05, 0.05),
        (-0.05, 0.05),
        (5-0.05, 5+0.05),
        (-0.025, 0.025),
        (-0.025, 0.025),
        (-0.0125, 0.0125),
    ])
    npts = len(points)
    intersections = np.zeros((npts, 10), dtype=float)
    nintersections = np.zeros(npts, dtype=int)
    nblocks = ceil(npts/threadsperblock)
    intersect_kernel[nblocks, threadsperblock](
        points, velocities, intersections, nintersections)
    # for i in range(npts):
    #     print(intersections[i, :nintersections[i]])
    np.testing.assert_allclose(intersections[:, :2], expected)
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_union_example2():
    parsed = parse_file(os.path.join(thisdir, 'union_example2.xml'))
    union1 = parsed[0]
    f = locate.LocateFuncFactory()
    devf_locate = f.render(union1)
    assert devf_locate(0,0,0) == location.inside
    assert devf_locate(0,0,0.1) == location.onborder
    assert devf_locate(0,0,-0.1) == location.onborder
    f = arrow_intersect.ArrowIntersectFuncFactory()
    devf_arrow_intersect = f.render(union1)
    ts = np.zeros(10)
    N = devf_arrow_intersect(0,0,0, 0,0,1., ts, 0)
    # print(ts[:N])
    np.testing.assert_allclose(ts[:N], [-0.1, 0.1])
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_union_example2_kernel():
    parsed = parse_file(os.path.join(thisdir, 'union_example2.xml'))
    union1 = parsed[0]
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
    ])
    velocities = np.array([
        (0.,0.,1.),
    ])
    expected = np.array([
        (-0.1, 0.1),
    ])
    npts = len(points)
    intersections = np.zeros((npts, 10), dtype=float)
    nintersections = np.zeros(npts, dtype=int)
    nblocks = ceil(npts/threadsperblock)
    intersect_kernel[nblocks, threadsperblock](
        points, velocities, intersections, nintersections)
    # for i in range(npts):
    #     print(intersections[i, :nintersections[i]])
    np.testing.assert_allclose(intersections[:, :2], expected)
    return


@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_difference_example1():
    parsed = parse_file(os.path.join(thisdir, 'difference_example1.xml'))
    difference = parsed[0]
    f = locate.LocateFuncFactory()
    devf_locate = f.render(difference)
    # test shape is sphere - cylinder: all points in cylinder should be outside shape
    assert devf_locate(0, 0, 0) == location.outside
    assert devf_locate(0, 0, 0.03) == location.outside
    assert devf_locate(0, 0, 0.049999) == location.outside
    assert devf_locate(0, 0, 0.05) == location.outside
    assert devf_locate(0, 0, 0.10) == location.outside

    # test that edges of the cylinder inside sphere are on-border
    assert devf_locate(0, 0.01, 0) == location.onborder
    assert devf_locate(0.01, 0, 0) == location.onborder
    assert devf_locate(0, 0.01, 0.025 * math.sin(math.acos(0.01 / 0.025))) == location.onborder
    assert devf_locate(0.01, 0, 0.025 * math.sin(math.acos(0.01 / 0.025))) == location.onborder

    assert devf_locate(0, 0.0249999, 0) == location.inside
    assert devf_locate(0, 0.025, 0) == location.onborder
    assert devf_locate(0, 0.0250001, 0) == location.outside

    f = arrow_intersect.ArrowIntersectFuncFactory()
    devf_arrow_intersect = f.render(difference)
    ts = np.zeros(10)
    # ray intersection through Z should miss entirely
    N = devf_arrow_intersect(0, 0, 0, 0, 0, 1.0, ts, 0)
    assert N == 0

    # ray intersection to +X should hit 4 times:
    # -X sphere edge, inner left and right side of inside hollow cylinder, then +X sphere edge
    ts = np.zeros(10)
    N = devf_arrow_intersect(0, 0, 0, 1.0, 0, 0, ts, 0)
    assert N == 4
    np.testing.assert_allclose(ts[:N], [-0.025, -0.01, 0.01, 0.025])
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_difference_example1_kernel():
    parsed = parse_file(os.path.join(thisdir, 'difference_example1.xml'))
    difference1 = parsed[0]
    f = locate.LocateFuncFactory()
    devf_locate = f.render(difference1)

    @cuda.jit
    def locate_kernel(points, locations):
        idx = cuda.grid(1)
        if idx < len(points):
            x, y, z = points[idx]
            locations[idx] = devf_locate(x, y, z)

    points = np.array([
        (0, 0, 0),
        (0, 0, 0.03),
        (0, 0, 0.049999),
        (0, 0, 0.05),
        (0, 0, 0.10),
        (0, 0.01, 0),
        (0.01, 0, 0),
        (0, 0.01, 0.025 * math.sin(math.acos(0.01 / 0.025))),
        (0.01, 0, 0.025 * math.sin(math.acos(0.01 / 0.025))),
        (0, 0.0249999, 0)
    ])
    expected = np.array([
        location.outside,
        location.outside,
        location.outside,
        location.outside,
        location.outside,
        location.onborder,
        location.onborder,
        location.onborder,
        location.onborder,
        location.inside
    ])
    N = len(points)
    locations = np.zeros(N, dtype=int)
    nblocks = ceil(N/threadsperblock)
    locate_kernel[nblocks, threadsperblock](points, locations)
    np.testing.assert_array_equal(locations, expected)
    # intersect
    f = arrow_intersect.ArrowIntersectFuncFactory()
    devf_intersect = f.render(difference1)
    @cuda.jit
    def intersect_kernel(points, velocities, intersections, nintersections):
        idx = cuda.grid(1)
        if idx < len(points):
            x, y, z = points[idx]
            vx, vy, vz = velocities[idx]
            n = devf_intersect(x, y, z, vx, vy, vz, intersections[idx], 0)
            nintersections[idx] = n
    points = np.array([
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        (0.005, 0.005, 0.),
        (-0.005, 0.005, 0.),
        (0., 0., -5.),
        (0., 0., 0.),
    ])
    velocities = np.array([
        (0., 0., 1.),
        (1., 0., 0.),
        (0., 1., 0.),
        (0., 0., 1.),
        (0., 0., 1.),
        (0., 0., 1.),
        (math.sqrt(2), math.sqrt(2), 0.),
    ])
    expected = np.array([
        (),
        (-0.025, -0.01, 0.01, 0.025),
        (-0.025, -0.01, 0.01, 0.025),
        (),
        (),
        (),
        (-0.5 * 0.025, -0.5 * 0.01, 0.5 * 0.01, 0.5 * 0.025),
    ])
    npts = len(points)
    intersections = np.zeros((npts, 10), dtype=float)
    nintersections = np.zeros(npts, dtype=int)
    nblocks = ceil(npts/threadsperblock)
    intersect_kernel[nblocks, threadsperblock](
        points, velocities, intersections, nintersections)
    for i in range(npts):
        np.testing.assert_allclose(intersections[i, :nintersections[i]], expected[i])
    return


@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_intersection_example1():
    parsed = parse_file(os.path.join(thisdir, 'intersection_example1.xml'))
    intersection = parsed[0]
    f = locate.LocateFuncFactory()
    devf_locate = f.render(intersection)
    assert devf_locate(0, 0, 0) == location.inside
    assert devf_locate(0, 0, 0.03) == location.outside
    assert devf_locate(0, 0, 0.025) == location.onborder
    assert devf_locate(0, 0, 0.05) == location.outside
    assert devf_locate(0, 0, 0.050001) == location.outside
    assert devf_locate(0, 0.00999, 0) == location.inside
    assert devf_locate(0, 0.01, 0) == location.onborder
    assert devf_locate(0, 0.010001, 0) == location.outside
    f = arrow_intersect.ArrowIntersectFuncFactory()
    devf_arrow_intersect = f.render(intersection)
    ts = np.zeros(10)

    # ray intersection to +Z should hit top and bottom of cylinder at sphere radius
    N = devf_arrow_intersect(0, 0, 0, 0, 0, 1., ts, 0)
    assert N == 2
    np.testing.assert_allclose(ts[:N], [-0.025, 0.025])

    # ray intersection to +X should hit internal cylinder
    ts = np.zeros(10)
    N = devf_arrow_intersect(0, 0, 0, 1., 0, 0., ts, 0)
    assert N == 2
    np.testing.assert_allclose(ts[:N], [-0.01, 0.01])
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_intersection_example1_kernel():
    parsed = parse_file(os.path.join(thisdir, 'intersection_example1.xml'))
    intersection = parsed[0]
    f = locate.LocateFuncFactory()
    devf_locate = f.render(intersection)

    @cuda.jit
    def locate_kernel(points, locations):
        idx = cuda.grid(1)
        if idx < len(points):
            x, y, z = points[idx]
            locations[idx] = devf_locate(x, y, z)

    points = np.array([
        (0, 0, 0),
        (0, 0, 0.03),
        (0, 0, 0.025),
        (0, 0, 0.05),
        (0, 0, 0.050001),
        (0, 0.00999, 0),
        (0, 0.01, 0),
        (0, 0.0100001, 0),
    ])
    expected = np.array([
        location.inside,
        location.outside,
        location.onborder,
        location.outside,
        location.outside,
        location.inside,
        location.onborder,
        location.outside,
    ])
    N = len(points)
    locations = np.zeros(N, dtype=int)
    nblocks = ceil(N / threadsperblock)
    locate_kernel[nblocks, threadsperblock](points, locations)
    np.testing.assert_array_equal(locations, expected)
    # intersect
    f = arrow_intersect.ArrowIntersectFuncFactory()
    devf_intersect = f.render(intersection)

    @cuda.jit
    def intersect_kernel(points, velocities, intersections, nintersections):
        idx = cuda.grid(1)
        if idx < len(points):
            x, y, z = points[idx]
            vx, vy, vz = velocities[idx]
            n = devf_intersect(x, y, z, vx, vy, vz, intersections[idx], 0)
            nintersections[idx] = n

    points = np.array([
        (0., 0., 0.),
        (0.001, 0.001, 0.),
        (0.005, 0.005, 0.),
        (-0.005, 0.005, 0.),
        (0., 0., -5.),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
    ])
    velocities = np.array([
        (0., 0., 1.),
        (0., 0., 1.),
        (0., 0., 1.),
        (0., 0., 1.),
        (0., 0., 1.),
        (1., 0., 0.),
        (0., 1., 0.),
        (math.sqrt(2), math.sqrt(2), 0.),
    ])
    t2 = math.sqrt(0.025**2 - 0.001**2 - 0.001**2)
    t3 = math.sqrt(0.025**2 - 0.005**2 - 0.005**2)
    expected = np.array([
        (-0.025, 0.025),
        (-t2, t2),
        (-t3, t3),
        (-t3, t3),
        (5 - 0.025, 5 + 0.025),
        (-0.01, 0.01),
        (-0.01, 0.01),
        (0.5 * -0.01, 0.5 * 0.01),
    ])
    npts = len(points)
    intersections = np.zeros((npts, 10), dtype=float)
    nintersections = np.zeros(npts, dtype=int)
    nblocks = ceil(npts / threadsperblock)
    intersect_kernel[nblocks, threadsperblock](
        points, velocities, intersections, nintersections)
    for i in range(npts):
        # print(i, intersections[i, :nintersections[i]])
        np.testing.assert_allclose(intersections[i, :nintersections[i]], expected[i])
    return


def main():
    # test_union_example2()
    # test_intersection_example1()
    test_intersection_example1_kernel()
    return

if __name__ == '__main__': main()

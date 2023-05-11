# Example code that would be generated automatically

import os, numpy as np, numba
from numba import cuda
from mcvine.acc._numba import xoroshiro128p_uniform_float32
from mcvine.acc import test
from mcvine.acc.geometry import arrow_intersect
from mcvine.acc.geometry.location import inside, onborder, outside
from mcvine.acc.geometry.arrow_intersect import max_intersections, insert_into_sorted_list

def createMethods_3(shapes):
    "methods for a list of non-overlapping shapes"
    assert len(shapes)==3
    funcs_list = [
        (
            arrow_intersect.locate_func_factory.render(shape),
            arrow_intersect.arrow_intersect_func_factory.render(shape),
        )
        for shape in shapes
    ]

    locate_0, intersect_0 = funcs_list[0]
    locate_1, intersect_1 = funcs_list[1]
    locate_2, intersect_2 = funcs_list[2]

    @cuda.jit(device=True)
    def _intersect_all(x,y,z, vx,vy,vz, ts, ts_):
        N = 0
        N_ = intersect_0(x,y,z, vx,vy,vz, ts_)
        for i in range(N_):
            N = insert_into_sorted_list(ts_[i], ts, N)
        N_ = intersect_1(x,y,z, vx,vy,vz, ts_)
        for i in range(N_):
            N = insert_into_sorted_list(ts_[i], ts, N)
        N_ = intersect_2(x,y,z, vx,vy,vz, ts_)
        for i in range(N_):
            N = insert_into_sorted_list(ts_[i], ts, N)
        return N

    @cuda.jit(device=True)
    def _forward_intersect_all(x,y,z, vx,vy,vz, ts, ts_):
        N = 0
        N_ = intersect_0(x,y,z, vx,vy,vz, ts_)
        for i in range(N_):
            t = ts_[i]
            if t>0:
                N = insert_into_sorted_list(t, ts, N)
        N_ = intersect_1(x,y,z, vx,vy,vz, ts_)
        for i in range(N_):
            t = ts_[i]
            if t>0:
                N = insert_into_sorted_list(t, ts, N)
        N_ = intersect_2(x,y,z, vx,vy,vz, ts_)
        for i in range(N_):
            t = ts_[i]
            if t>0:
                N = insert_into_sorted_list(t, ts, N)
        return N

    @cuda.jit(device=True)
    def find_shape_containing_point(x,y,z):
        if locate_0(x,y,z) == inside:
            return 0
        if locate_1(x,y,z) == inside:
            return 1
        if locate_2(x,y,z) == inside:
            return 2
        return -1

    @cuda.jit(device=True)
    def is_onborder(x,y,z):
        "check if the point is on the border of any of the shapes"
        if locate_0(x,y,z) == onborder:
            return True
        if locate_1(x,y,z) == onborder:
            return True
        if locate_2(x,y,z) == onborder:
            return True
        return False

    if test.USE_CUDASIM:
        @cuda.jit(device=True)
        def intersect_all(x,y,z, vx,vy,vz, ts):
            ts_ = np.zeros(max_intersections, dtype=float)
            return _intersect_all(x,y,z, vx,vy,vz, ts, ts_)
        @cuda.jit(device=True)
        def forward_intersect_all(x,y,z, vx,vy,vz, ts):
            ts_ = np.zeros(max_intersections, dtype=float)
            return _forward_intersect_all(x,y,z, vx,vy,vz, ts, ts_)
    else:
        @cuda.jit(device=True)
        def intersect_all(x,y,z, vx,vy,vz, ts):
            ts_ = cuda.local.array(max_intersections, dtype=numba.float64)
            return _intersect_all(x,y,z, vx,vy,vz, ts, ts_)
        @cuda.jit(device=True)
        def forward_intersect_all(x,y,z, vx,vy,vz, ts):
            ts_ = cuda.local.array(max_intersections, dtype=numba.float64)
            return _forward_intersect_all(x,y,z, vx,vy,vz, ts, ts_)
    return dict(
        intersect_all = intersect_all,
        forward_intersect_all = forward_intersect_all,
        find_shape_containing_point = find_shape_containing_point,
        is_onborder = is_onborder,
    )

def createUnionLocateMethod_3(shapes):
    assert len(shapes)==3
    locates = [
        arrow_intersect.locate_func_factory.render(shape)
        for shape in shapes
    ]
    locate_0 = locates[0]
    locate_1 = locates[1]
    locate_2 = locates[2]

    @cuda.jit(device=True)
    def locate(x,y,z):
        loc0 = locate_0(x,y,z)
        loc1 = locate_1(x,y,z)
        loc2 = locate_2(x,y,z)
        if loc0 == inside: return inside
        if loc1 == inside: return inside
        if loc2 == inside: return inside
        if loc0 == onborder: return onborder
        if loc1 == onborder: return onborder
        if loc2 == onborder: return onborder
        return outside

    return locate

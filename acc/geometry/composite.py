def makeMethods(shapes):
    mod = makeModule(shapes)
    import imp
    m = imp.load_source('mod', mod)
    return m.createKernelMethods(composite)

def makeModule(shapes):
    "make cuda device methods for shapes"
    from .._numba import coder
    nshapes = len(shapes)
    modulepath = coder.getModule("shapes", nshapes)
    if os.path.exists(modulepath):
        return modulepath
    indent = 4*' '
    add_indent = lambda lines, n: [indent*n+l for l in lines]
    element_scatter_method_defs = [
        f'scatter_{ik}, scattering_coeff_{ik}, absorb_{ik}, absorption_coeff_{ik} = kernel_funcs_list[{ik}]'
        for ik in range(nkernels)
    ]


# Example code
import os, numpy as np
from numba import cuda
from mcvine.acc._numba import xoroshiro128p_uniform_float32
from mcvine.acc import test

def createMethods_3(shapes):
    assert len(shapes)==3
    Nshapes = len(shapes)
    from mcvine.acc.geometry import arrow_intersect
    from mcvine.acc.geometry.location import inside, onborder
    from mcvine.acc.geometry.arrow_intersect import max_intersections, insert_into_sorted_list
    funcs_list = []
    for shape in shapes:
        intersect = arrow_intersect.arrow_intersect_func_factory.render(shape)
        locate = arrow_intersect.locate_func_factory.render(shape)
        funcs = locate, intersect
        funcs_list.append(funcs)
        continue

    locate_0, intersect_0 = funcs_list[0]
    locate_1, intersect_1 = funcs_list[1]
    locate_2, intersect_2 = funcs_list[2]

    @cuda.jit(device=True)
    def _forward_intersect_all(x,y,z, vx,vy,vz, ts, ts0, ts1, ts2):
        N0 = intersect_0(x,y,z, vx,vy,vz, ts0)
        N1 = intersect_1(x,y,z, vx,vy,vz, ts1)
        N2 = intersect_2(x,y,z, vx,vy,vz, ts2)
        N = 0
        for i in range(N0):
            t = ts0[i]
            if t>0:
                N = insert_into_sorted_list(t, ts, N)
        for i in range(N1):
            t = ts1[i]
            if t>0:
                N = insert_into_sorted_list(t, ts, N)
        for i in range(N2):
            t = ts2[i]
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

    @cuda.jit(device=True)
    def find_1st_hit(start, direction):
        raise NotImplementedError

    if test.USE_CUDASIM:
        @cuda.jit(device=True)
        def forward_intersect_all(x,y,z, vx,vy,vz, ts):
            ts0 = np.zeros(max_intersections, dtype=float)
            ts1 = np.zeros(max_intersections, dtype=float)
            ts2 = np.zeros(max_intersections, dtype=float)
            return _forward_intersect_all(x,y,z, vx,vy,vz, ts, ts0, ts1, ts2)
    else:
        @cuda.jit(device=True)
        def forward_intersect_all(x,y,z, vx,vy,vz, ts):
            ts0 = cuda.local.array(max_intersections, dtype=numba.float64)
            ts1 = cuda.local.array(max_intersections, dtype=numba.float64)
            ts2 = cuda.local.array(max_intersections, dtype=numba.float64)
            return _forward_intersect_all(x,y,z, vx,vy,vz, ts, ts0, ts1, ts2)
    return dict(
        forward_intersect_all = forward_intersect_all,
        find_shape_containing_point = find_shape_containing_point,
        find_1st_hit = find_1st_hit,
    )

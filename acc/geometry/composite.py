from mcvine.acc.geometry.arrow_intersect import max_intersections

def make_find_1st_hit(forward_intersect_all, is_onborder, find_shape_containing_point):
    @cuda.jit(device=True)
    def _find_1st_hit(x,y,z, vx,vy,vz, ts):
        nIntersections = forward_intersect_all(x,y,z, vx,vy,vz, ts)
        # we have two cases
        # case 1
        #   shape1     |  vacuum | shape2
        #        start-|---------|---> 
        # case 2
        #   vacuum     |  shape1
        #        start-|-----------> 
        # case 1: there will be odd number of intersections
        # case 2: there will be even number of intersections
        # we just need to determine which one of the above two cases is true,
        # case1:
        if nIntersections % 2 == 1:
            ret = find_shape_containing_point(
                x + ts[0]/2.*vx,
                y + ts[0]/2.*vy,
                z + ts[0]/2.*vz,
            )
        else :
            # case2:
            # no intersection
            if nIntersections==0:
                return -1
        # at least two
            midt = (ts[0]+ts[1])/2.
            ret = find_shape_containing_point(
                x + midt*vx,
                y + midt*vy,
                z + midt*vz,
            )
            # let us determine if the start is on border
            isonborder = is_onborder(x,y,z)
            # If start is not on border of any shape, it would be easier.
            if (not isonborder): return ret

            # on border. that is a bit more difficult.
            # we need to go over all intersection pairs, and find the first pair
            # whose midlle point is insde a shape. That shape containing
            # the middle point is the target.
            previous = 0.0
            for point_index in range(nIntersections):
                now = ts[point_index]
                midt = (previous+now)/2
                ret = find_shape_containing_point(
                    x + midt*vx,
                    y + midt*vy,
                    z + midt*vz,
                )
                if ret>=0: return ret
                previous = now
        return -1
    if test.USE_CUDASIM:
        @cuda.jit(device=True)
        def find_1st_hit(x,y,z, vx,vy,vz):
            ts = np.zeros(max_intersections, dtype=float)
            return _find_1st_hit(x,y,z, vx,vy,vz, ts)
    else:
        @cuda.jit(device=True)
        def find_1st_hit(x,y,z, vx,vy,vz):
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            return _find_1st_hit(x,y,z, vx,vy,vz, ts)
    return find_1st_hit


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
        is_onborder = is_onborder,
    )

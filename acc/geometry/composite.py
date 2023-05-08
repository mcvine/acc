import os, numpy as np, numba
from numba import cuda
from mcvine.acc.geometry.arrow_intersect import max_intersections
from mcvine.acc import test

def make_find_1st_hit(forward_intersect_all, is_onborder, find_shape_containing_point, **kwds):
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
            return find_shape_containing_point(
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



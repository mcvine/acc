import os, numpy as np, numba
from numba import cuda
from mcvine.acc.geometry.arrow_intersect import max_intersections
from mcvine.acc import test
from .._numba import coder

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

def makeModule(N, overwrite=False):
    "make cuda device methods for composite with N elements"
    from .._numba import coder
    modulepath = coder.getModule("composite_shape", N)
    if os.path.exists(modulepath) and not overwrite:
        return modulepath
    indent = 4*' '
    lines = _create_createMethods(N, indent)
    with open(modulepath, 'wt') as ostream:
        ostream.write("\n".join(lines))

def _create_createMethods(N, indent=4*' '):
    header = f"""assert len(shapes)=={N}
funcs_list = [
    (
    arrow_intersect.locate_func_factory.render(shape),
    arrow_intersect.arrow_intersect_func_factory.render(shape),
    )
    for shape in shapes
]
""".splitlines()
    funcs_loop = coder.unrollLoop(
        N = N,
        indent = '',
        in_loop     = ["locate_{i}, intersect_{i} = funcs_list[{i}]"],
    )
    end = """
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
    """.splitlines()
    body = (
        header + funcs_loop
        + _create__intersect_all(N, indent)
        + _create__forward_intersect_all(N, indent)
        + _create_find_shape_containing_point(N, indent)
        + _create_is_onborder(N, indent)
        + end
    )
    add_indent = lambda lines, n: [indent*n+l for l in lines]
    return ["def createMethods(shapes):"] + add_indent(body, 1)


def _create__intersect_all(N, indent=4*' '):
    header = [
        "@cuda.jit(device=True)",
        "def _intersect_all(x,y,z, vx,vy,vz, ts, ts_):",
    ]
    loop = coder.unrollLoop(
        N = N,
        indent = indent,
        before_loop = ["N=0"],
        in_loop     = [
            "N_ = intersect_{i}(x,y,z, vx,vy,vz, ts_)",
            "for i in range(N_):",
            indent + "N = insert_into_sorted_list(ts_[i], ts, N)",
        ],
        after_loop  = ["return N"]
    )
    return header + loop

def _create__forward_intersect_all(N, indent=4*' '):
    header = [
        "@cuda.jit(device=True)",
        "def _forward_intersect_all(x,y,z, vx,vy,vz, ts, ts_):",
    ]
    loop = coder.unrollLoop(
        N = N,
        indent = indent,
        before_loop = ["N=0"],
        in_loop     = [
            "N_ = intersect_{i}(x,y,z, vx,vy,vz, ts_)",
            "for i in range(N_):",
            indent + "t = ts_[i]",
            indent + "if t>0:",
            indent*2 + "N = insert_into_sorted_list(t, ts, N)",
        ],
        after_loop  = ["return N"]
    )
    return header + loop

def _create_find_shape_containing_point(N, indent=4*' '):
    header = [
        "@cuda.jit(device=True)",
        "def find_shape_containing_point(x,y,z):",
    ]
    loop = coder.unrollLoop(
        N = N,
        indent = indent,
        in_loop     = [
            "if locate_{i}(x,y,z) == inside:",
            indent + "return {i}",
        ],
        after_loop  = ["return -1"]
    )
    return header + loop

def _create_is_onborder(N, indent=4*' '):
    header = [
        "@cuda.jit(device=True)",
        "def is_onborder(x,y,z):",
    ]
    loop = coder.unrollLoop(
        N = N,
        indent = indent,
        in_loop     = [
            "if locate_{i}(x,y,z) == onborder:",
            indent + "return True",
        ],
        after_loop  = ["return False"]
    )
    return header + loop

module_code_template = """
import os, numpy as np, numba
from numba import cuda
from mcvine.acc._numba import xoroshiro128p_uniform_float32
from mcvine.acc import test
from mcvine.acc.geometry import arrow_intersect
from mcvine.acc.geometry.location import inside, onborder, outside
from mcvine.acc.geometry.arrow_intersect import max_intersections, insert_into_sorted_list

"""

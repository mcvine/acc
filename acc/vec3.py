import math, numpy as np, numba
from numba import cuda
from mcvine.acc.config import get_numba_floattype, get_numpy_floattype
NB_FLOAT = get_numba_floattype()
from . import test

@cuda.jit(device=True, inline=True)
def dot(v1, v2):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

@cuda.jit(device=True, inline=True)
def cross(v1, v2, vout):
    vout[0] = v1[1]*v2[2]-v1[2]*v2[1]
    vout[1] = v1[2]*v2[0]-v1[0]*v2[2]
    vout[2] = v1[0]*v2[1]-v1[1]*v2[0]
    return

@cuda.jit(device=True, inline=True)
def length(v):
    return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])


@cuda.jit(device=True, inline=True)
def length2(v):
    """
    Returns squared magnitude of v
    """
    return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]

@cuda.jit(device=True, inline=True)
def normalize(v):
    l = length(v)
    scale(v, 1.0/l)
    return

@cuda.jit(device=True, inline=True)
def scale(v, s):
    v[0]*=s
    v[1]*=s
    v[2]*=s
    return

@cuda.jit(device=True, inline=True)
def copy(v1, v2):
    v2[0]=v1[0]
    v2[1]=v1[1]
    v2[2]=v1[2]
    return

@cuda.jit(device=True, inline=True)
def add(v1, v2, v3):
    v3[0]=v1[0]+v2[0]
    v3[1]=v1[1]+v2[1]
    v3[2]=v1[2]+v2[2]
    return

@cuda.jit(device=True, inline=True)
def subtract(v1, v2, v3):
    v3[0]=v1[0]-v2[0]
    v3[1]=v1[1]-v2[1]
    v3[2]=v1[2]-v2[2]
    return

@cuda.jit(device=True, inline=True)
def mXv(mat, vec, outvec):
    for i in range(3):
        outvec[i] = mat[i, 0]*vec[0] + mat[i, 1]*vec[1] + mat[i, 2]*vec[2]
    return

# cs_rot * ( r - cs_pos )
@cuda.jit(device=True, inline=True)
def _abs2rel(vec, rotmat, offset, vecout, tmp):
    subtract(vec, offset, tmp)
    mXv(rotmat, tmp, vecout)
    return

@cuda.jit(device=True)
def _rotate(v, c, angle, c1, v_pl, v_pp, e3, v_pp_r, epsilon=1e-7):
    #
    # normalize c
    # v_pl = (v dot c) * c      (parallel)
    # v_pp = v - v_pl           (perpendicular)
    # e_pp = v_pp / |v_pp|
    # t = (e_pp * c)     (cross product of c * v)
    # v_pp_r = (e_pp * cos(theta) - t * sin(theta))*|v_pp| (rotated v_pp)
    # v_r = v_pp_r + v_pl  (rotated v)
    #
    copy(c, c1)
    normalize(c1)

    scale_factor = dot(v, c1)
    copy(c1, v_pl)
    scale(v_pl, scale_factor)
    if abs(1-length(v_pl)/length(v))<epsilon:
        return
    subtract(v, v_pl, v_pp)
    v_pp_len = length(v_pp)
    scale(v_pp, 1./v_pp_len) # v_pp is now e_pp
    cross(v_pp, c1, e3)
    scale(v_pp, math.cos(angle))
    scale(e3, math.sin(angle))
    subtract(v_pp, e3, v_pp_r)
    scale(v_pp_r, v_pp_len)
    add(v_pp_r, v_pl, v)
    return

if test.USE_CUDASIM:
    @cuda.jit(device=True, inline=True)
    def rotate(v, c, angle, epsilon=1e-7):
        c1 = np.zeros(3, dtype=float)
        v_pl = np.zeros(3, dtype=float)
        v_pp = np.zeros(3, dtype=float)
        e3 = np.zeros(3, dtype=float)
        v_pp_r = np.zeros(3, dtype=float)
        return _rotate(
            v, c, angle, c1, v_pl, v_pp, e3, v_pp_r, epsilon)

    @cuda.jit(device=True, inline=True)
    def abs2rel(vec, rotmat, offset, vecout):
        tmp = np.zeros(3, dtype=float)
        _abs2rel(vec, rotmat, offset, vecout, tmp)
        return
else:
    @cuda.jit(device=True, inline=True)
    def rotate(v, c, angle, epsilon=1e-7):
        c1 = cuda.local.array(3, dtype=numba.float64)
        v_pl = cuda.local.array(3, dtype=numba.float64)
        v_pp = cuda.local.array(3, dtype=numba.float64)
        e3 = cuda.local.array(3, dtype=numba.float64)
        v_pp_r = cuda.local.array(3, dtype=numba.float64)
        return _rotate(
            v, c, angle, c1, v_pl, v_pp, e3, v_pp_r, epsilon)

    @cuda.jit(device=True, inline=True)
    def abs2rel(vec, rotmat, offset, vecout):
        tmp = cuda.local.array(3, dtype=NB_FLOAT)
        _abs2rel(vec, rotmat, offset, vecout, tmp)
        return

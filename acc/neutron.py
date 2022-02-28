#
from numba import cuda
from . import vec3

@cuda.jit(device=True, inline=True)
def abs2rel(r, v, rotmat, offset, rtmp, vtmp):
    vec3.copy(r, rtmp); vec3.copy(v, vtmp)
    vec3.abs2rel(rtmp, rotmat, offset, r)
    vec3.mXv(rotmat, vtmp, v)

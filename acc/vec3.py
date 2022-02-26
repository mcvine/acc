import math
from numba import cuda

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
def add(v1, v2, v3):
    v3[0]=v1[0]+v2[0]
    v3[1]=v1[1]+v2[1]
    v3[2]=v1[2]+v2[2]
    return

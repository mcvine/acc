import os, time
import numpy as np, math, numba
from numba import cuda
# import cupy as cp
from . import epsilon, location

@cuda.jit(device=True, inline=True)
def cu_device_locate_wrt_sphere(x,y,z, R):
    dist2 = x*x+y*y+z*z
    if dist2>(R+epsilon)*(R+epsilon): return location.outside
    elif dist2<(R-epsilon)*(R-epsilon): return location.inside
    return location.onborder

@cuda.jit(device=True, inline=True)
def cu_device_intersect_sphere(x,y,z, vx,vy,vz, R):
    t1 = np.nan; t2 = np.nan
    # not implemented
    return t1, t2


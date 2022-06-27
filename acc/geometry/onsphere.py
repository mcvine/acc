import os, time
import numpy as np, math, numba
from numba import cuda
# import cupy as cp
from . import epsilon

@cuda.jit(device=True, inline=True)
def cu_device_inside_sphere(x,y,z, R):
    return x*x+y*y+z*z<R*R

@cuda.jit(device=True, inline=True)
def cu_device_onborderof_sphere(x,y,z, R):
    dist2 = x*x+y*y+z*z
    return dist2<(R+epsilon)*(R+epsilon) and dist2>(R-epsilon)*(R-epsilon)

@cuda.jit(device=True, inline=True)
def cu_device_intersect_sphere(x,y,z, vx,vy,vz, R):
    t1 = np.nan; t2 = np.nan
    # not implemented
    return t1, t2


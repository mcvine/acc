import os, time
import numpy as np, math, numba
from numba import cuda
from . import epsilon, location

@cuda.jit(device=True, inline=True)
def cu_device_locate_wrt_cylinder(x,y,z, R, H):
    if ( abs(z)-H/2. > epsilon ) :
        return location.outside

    r2 = x*x+y*y
    if r2 > (R+epsilon)*(R+epsilon):
        return location.outside

    if H/2.-abs(z) > epsilon and r2 < (R-epsilon)*(R-epsilon):
        return location.inside
    return location.onborder

@cuda.jit(device=True, inline=True)
def cu_device_intersect_cylinder(x,y,z, vx,vy,vz, R, H):
    t1 = np.nan; t2 = np.nan
    # not implemented
    return t1, t2


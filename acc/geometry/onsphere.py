import os, time
import numpy as np, math, numba
from numba import cuda
# import cupy as cp

@cuda.jit(device=True, inline=True)
def cu_device_intersect_sphere(x,y,z, vx,vy,vz, R):
    t1 = np.nan; t2 = np.nan
    # not implemented
    return t1, t2


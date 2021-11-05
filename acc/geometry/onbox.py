import os, time
import numpy as np, math, numba
from numba import cuda
# import cupy as cp

epsilon = 1e-15

@cuda.jit(
    """float32(float32, float32, float32, float32, float32, float32,
               float32, float32)""",
    device=True, inline=True)
def cu_device_intersect_rectangle(rx, ry, rz, vx, vy, vz, X, Y):
    t = - rz / vz
    r1x = rx + vx * t
    r1y = ry + vy * t
    if math.fabs(r1x) > X / 2 or math.fabs(r1y) > Y / 2:
        t = np.nan
    return t

@cuda.jit(device=True, inline=True)
def cu_device_update_intersections(t1, t2, t):
    """update intersections. t1, t2 are the intersections to update and return
    """
    # t is nan
    if not math.isfinite(t): return t1, t2
    # t1 is nan (it means t2 is nan too)
    if not math.isfinite(t1): return t, t2
    # t1 is good. t2 is nan
    if not math.isfinite(t2):
        if t>t1: return t1, t
        else: return t, t1
    # t1, t2, t are all finite
    if t < t1: return t, t2
    elif t < t2: return t1, t2
    return t1, t

@cuda.jit(device=True, inline=True)
def cu_device_intersect_box(x,y,z, vx,vy,vz, X, Y, Z):
    t1 = np.nan; t2 = np.nan
    if (vz!=0) :
        t = cu_device_intersect_rectangle(x,y,z-Z/2, vx,vy,vz, X, Y)
        t1, t2 = cu_device_update_intersections(t1, t2, t)
        t = cu_device_intersect_rectangle(x,y,z+Z/2, vx,vy,vz, X, Y)
        t1, t2 = cu_device_update_intersections(t1, t2, t)
    if (vx!=0) :
        t = cu_device_intersect_rectangle(y,z,x-X/2, vy,vz,vx, Y, Z)
        t1, t2 = cu_device_update_intersections(t1, t2, t)
        t = cu_device_intersect_rectangle(y,z,x+X/2, vy,vz,vx, Y, Z)
        t1, t2 = cu_device_update_intersections(t1, t2, t)
    if (vy!=0) :
        t = cu_device_intersect_rectangle(z,x,y-Y/2, vz,vx,vy, Z, X);
        t1, t2 = cu_device_update_intersections(t1, t2, t)
        t = cu_device_intersect_rectangle(z,x,y+Y/2, vz,vx,vy, Z, X);
        t1, t2 = cu_device_update_intersections(t1, t2, t)
    return t1, t2

@numba.guvectorize(
    ["float32[:], float32[:], float32[:], float32[:], float32[:], float32[:], float32, float32, float32, float32[:], float32[:, :]"],
    '(N), (N), (N), (N), (N), (N), (), (), (), (n) -> (N, n)',
    target='cuda'
)
def cu_intersect_box(x, y, z, vx, vy, vz, sx, sy, sz, _, ts):
    "_ array is only used to inform the dimension of output array ts"
    N, = x.shape
    n, = _.shape
    assert n==2
    for i in range(N):
        t1, t2 = cu_device_intersect_box(x[i], y[i], z[i], vx[i], vy[i], vz[i], sx, sy, sz)
        ts[i, 0] = t1
        ts[i, 1] = t2
    return ts

def test_cu_device_update_intersections():
    # only works when cuda jit is commented out
    assert cu_device_update_intersections(np.nan, np.nan, 3.) == (3.0, np.nan)
    assert cu_device_update_intersections(1., np.nan, 3.) == (1., 3.0)
    return

def test_cu_device_intersect_box():
    # only works when cuda jit is commented out
    assert cu_device_intersect_box(0,0,0, 0.,0.,1., 0.02, 0.02, 0.02) == (-0.01, 0.01)
    assert cu_device_intersect_box(0,0,0, 1.,1.,0., 0.02, 0.02, 0.02) == (-0.01, 0.01)
    assert cu_device_intersect_box(0,0,0, 1.,1.,1., 0.02, 0.02, 0.02) == (-0.01, 0.01)
    assert cu_device_intersect_box(0,0,0, 0.,0.,1., 0.02, 0.02, 0.04) == (-0.02, 0.02)
    return

import os, time
from numba import cuda
import numpy as np
import math
import numba
import cupy as cp

epsilon = 1e-15

@cuda.jit(
    """float32(float32, float32, float32, float32, float32, float32,
               float32, float32)""",
    device=True, inline=True)
def cu_device_intersect_rectangle(rx, ry, rz, vx, vy, vz, X, Y):
    t_pos = (0 - rz) / vz
    r1x = rx + vx * t_pos
    r1y = ry + vy * t_pos
    bad = math.fabs(r1x) > X / 2 or math.fabs(r1y) > Y / 2
    if bad:
        t_pos= np.nan
    return t_pos

@cuda.jit(device=True, inline=True)
def update_intercepts(t1, t2, t):
    """update intercepts. t1, t2 are the intercepts to update and return
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

def test_update_intercepts():
    # only works when cuda jit is commented out
    assert update_intercepts(np.nan, np.nan, 3.) == (3.0, np.nan)
    assert update_intercepts(1., np.nan, 3.) == (1., 3.0)
    return

@cuda.jit(device=True, inline=True)
def cu_device_intersect_box(x,y,z, vx,vy,vz, X, Y, Z):
    t1 = np.nan; t2 = np.nan
    if (vz!=0) :
        t = cu_device_intersect_rectangle(x,y,z-Z/2, vx,vy,vz, X, Y)
        t1, t2 = update_intercepts(t1, t2, t)
        t = cu_device_intersect_rectangle(x,y,z+Z/2, vx,vy,vz, X, Y)
        t1, t2 = update_intercepts(t1, t2, t)
    if (vx!=0) :
        t = cu_device_intersect_rectangle(y,z,x-X/2, vy,vz,vx, Y, Z)
        t1, t2 = update_intercepts(t1, t2, t)
        t = cu_device_intersect_rectangle(y,z,x+X/2, vy,vz,vx, Y, Z)
        t1, t2 = update_intercepts(t1, t2, t)
    if (vy!=0) :
        t = cu_device_intersect_rectangle(z,x,y-Y/2, vz,vx,vy, Z, X);
        t1, t2 = update_intercepts(t1, t2, t)
        t = cu_device_intersect_rectangle(z,x,y+Y/2, vz,vx,vy, Z, X);
        t1, t2 = update_intercepts(t1, t2, t)
    return t1, t2

def test_cu_device_intersect_box():
    # only works when cuda jit is commented out
    assert cu_device_intersect_box(0,0,0, 0.,0.,1., 0.02, 0.02, 0.02) == (-0.01, 0.01)
    assert cu_device_intersect_box(0,0,0, 1.,1.,0., 0.02, 0.02, 0.02) == (-0.01, 0.01)
    assert cu_device_intersect_box(0,0,0, 1.,1.,1., 0.02, 0.02, 0.02) == (-0.01, 0.01)
    assert cu_device_intersect_box(0,0,0, 0.,0.,1., 0.02, 0.02, 0.04) == (-0.02, 0.02)
    return

@numba.guvectorize(
    ["float32[:], float32[:], float32[:], float32[:], float32[:], float32[:], float32, float32, float32, float32[:]"],
    '(N), (N), (N), (N), (N), (N), (), (), () -> (N)',
    target='cuda'
)
def cu_intersect_box(x, y, z, vx, vy, vz, sx, sy, sz, t):
    N, = x.shape
    for i in range(N):
        t1, t2 = cu_device_intersect_box(x[i], y[i], z[i], vx[i], vy[i], vz[i], sx, sy, sz)
        t[i] = t1
    return t

def main():
    N = int(1e7)
    x_np = np.arange(-0.02, 0.02, 0.04/N, dtype='float32')
    y_np = np.arange(-0.02, 0.02, 0.04/N, dtype='float32')
    z_np = np.ones(x_np.size, dtype='float32')*(-.05)
    vx_np = np.zeros(x_np.size, dtype='float32')
    vy_np = np.zeros(x_np.size, dtype='float32')
    vz_np = np.ones(x_np.size, dtype='float32')
    x_cu = cuda.to_device(x_np)
    y_cu = cuda.to_device(y_np)
    z_cu = cuda.to_device(z_np)
    vx_cu = cuda.to_device(vx_np)
    vy_cu = cuda.to_device(vy_np)
    vz_cu = cuda.to_device(vz_np)
    print("starting")
    t1 = time.time()
    sx = sy = sz = 0.05
    cu_t = cu_intersect_box(
        x_cu, y_cu, z_cu, vx_cu, vy_cu, vz_cu,
        sx, sy, sz
    )
    t2 = time.time()
    print("done calculating")
    print(t2-t1)
    t = cu_t.copy_to_host()
    print(t[:10])
    return

def test():
    test_update_intercepts()
    test_cu_device_intersect_box()
    return

if __name__ == '__main__': main()

import os, time
import numpy as np, math, numba
from numba import cuda
# import cupy as cp

epsilon = 1e-15


@cuda.jit(
    """float32(float32, float32, float32, float32, float32, float32,
               float32, float32)""",
    device=True, inline=True)
def cu_device_intersectCylinderSide(x, y, z, vx, vy, vz, r, h):
    a = vx*vx + vy*vy
    b = x * vx + y * vy
    c = x * x + y * y - r * r
    k = b * b - a * c
    hh = h / 2

    if (k < 0):
        t1, t2=np.nan, np.nan
        return t1, t2
    if k == 0:
        t1, t2 = -b / a, -b / a
        if abs(z + vz * t1) > hh:
            t1, t2 =np.nan, np.nan
        return t1,t2

    k = np.sqrt(k)
    t1= (-b + k) / a
    if abs(z+vz * t1) > hh:
        t1 = np.nan

    t2 = (-b - k) / a
    if abs(z + vz * t2) > hh:
        t2 = np.nan
    return t1, t2


def cu_device_intersect_CylinderTopBottom(x, y, z, vx, vy, vz, r, h):
    hh = h / 2
    r2 = r * r

    t1 = (hh - z) / vz
    x1 = x + vx * t1
    y1 = y + vy * t1
    if x1 * x1 + y1 * y1 > r2:
        t1 = np.nan

    t2 = (-hh - z) / vz
    x1 = x + vx * t2
    y1 = y + vy * t2
    if x1 * x1 + y1 * y1 > r2:
        t2 = np.nan
    return t1, t2

@cuda.jit(device=True, inline=True)
def cu_device_update_intersections(t1, t2,t):
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
def cu_device_intersect_cylinder(x, y, z, vx, vy, vz, r, h):
    t1 = np.nan
    t2 = np.nan

    t1_side, t2_side = cu_device_intersectCylinderSide(x, y, z, vx, vy, vz, r, h)
    t1_topB, t2_topB = cu_device_intersect_CylinderTopBottom(x, y, z, vx, vy, vz, r, h)
    t1, t2 = cu_device_update_intersections(t1, t2,t1_side)
    t1, t2 = cu_device_update_intersections(t1, t2, t2_side)
    t1, t2 = cu_device_update_intersections(t1, t2, t1_topB)
    t1, t2 = cu_device_update_intersections(t1, t2, t2_topB)

    return t1, t2

@cuda.jit(device=True, inline=True)
def cu_device_update_intersections_multiple(t_intersections):
    t1 = np.nan
    t2 = np.nan

    for t in t_intersections:
        if math.isfinite(t):
            if math.finite(t1):
                t2 = t
                if t1>t2:
                    t1, t2 = t2, t1
            else:
                t1 =t

    return t1, t2

@cuda.jit(device=True, inline=True)
def cu_device_intersect_cylinder(x, y, z, vx, vy, vz, r, h):
    t_intersections = cuda.local.array(4, numba.float64)

    t1_side, t2_side = cu_device_intersectCylinderSide(x, y, z, vx, vy, vz, r, h)
    t1_topB, t2_topB = cu_device_intersect_CylinderTopBottom(x, y, z, vx, vy, vz, r, h)
    t_intersections[0] = t1_side
    t_intersections[1] = t1_side
    t_intersections[2] = t1_topB
    t_intersections[3] = t2_topB
    t1, t2 = cu_device_update_intersections_multiple(t_intersections)

    return t1, t2

@numba.guvectorize(
    ["float32[:], float32[:], float32[:], float32[:], float32[:], float32[:], float32, float32, float32, float32[:], float32[:, :]"],
    '(N), (N), (N), (N), (N), (N), (), (), (), (n) -> (N, n)',
    target='cuda'
)
def cu_intersect_cylinder(x, y, z, vx, vy, vz, r,h, _, ts):
    "_ array is only used to inform the dimension of output array ts"
    N, = x.shape
    n, = _.shape
    assert n==2
    for i in range(N):
        t1, t2 = cu_device_intersect_cylinder(x[i], y[i], z[i], vx[i], vy[i], vz[i], r,h)
        ts[i, 0] = t1
        ts[i, 1] = t2
    return ts

def test_cu_device_update_intersections():
    # only works when cuda jit is commented out
    assert cu_device_update_intersections(np.nan, np.nan, 3.) == (3.0, np.nan)
    assert cu_device_update_intersections(1., np.nan, 3.) == (1., 3.0)
    return

def test_cu_device_intersect_cylinder():
    # only works when cuda jit is commented out
    assert cu_device_intersect_cylinder(0,0,-5, 0.,0.,1., 1,2) == (4, 6)
    assert cu_device_intersect_cylinder(0,0,-5, 0,0,1., 1,1) == (4.5, 5.5)
    assert cu_device_intersect_cylinder(-5,0,0, 1.,0.,0., 1.,1.) == (4, 6)
    assert cu_device_intersect_cylinder(0.22,0.35,-5., 0.,0.,1., 1,1) == (4.5, 5.5)
    return

def test_cu_intersect_cylinder():
    N = int(1e7)
    x_np = np.arange(-0.02, 0.02, 0.04/N, dtype='float32')
    y_np = np.arange(-0.02, 0.02, 0.04/N, dtype='float32')
    z_np = np.ones(x_np.size, dtype='float32')*(-.05)
    vx_np = np.zeros(x_np.size, dtype='float32')
    vy_np = np.zeros(x_np.size, dtype='float32')
    vz_np = np.ones(x_np.size, dtype='float32')
    ts_dim_np = np.zeros(2, dtype='float32')
    x_cu = cuda.to_device(x_np)
    y_cu = cuda.to_device(y_np)
    z_cu = cuda.to_device(z_np)
    vx_cu = cuda.to_device(vx_np)
    vy_cu = cuda.to_device(vy_np)
    vz_cu = cuda.to_device(vz_np)
    ts_dim_cu = cuda.to_device(ts_dim_np)
    print("starting")
    t1 = time.time()
    h = r = 0.05
    ts_cu = cu_intersect_cylinder(
        x_cu, y_cu, z_cu, vx_cu, vy_cu, vz_cu,
        r, h,
        ts_dim_cu
    )
    t2 = time.time()
    print("done calculating")
    print(t2-t1)
    ts = ts_cu.copy_to_host()
    print(ts[:10])
    return

def main():
    # test_cu_device_update_intersections()
    # test_cu_device_intersect_box()
    test_cu_intersect_cylinder()
    return

if __name__ == '__main__': main()
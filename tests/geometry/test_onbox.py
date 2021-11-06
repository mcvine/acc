#!/usr/bin/env python

import pytest, os
from mcvine.acc import test

import os, numpy as np, time
from numba import cuda
from mcvine.acc.geometry import onbox

# device functions can be tested with CUDASIM only
@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_cu_device_update_intersections():
    # only works when cuda jit is commented out
    assert onbox.cu_device_update_intersections(np.nan, np.nan, 3.) == (3.0, np.nan)
    assert onbox.cu_device_update_intersections(1., np.nan, 3.) == (1., 3.0)
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_cu_device_intersect_box():
    # only works when cuda jit is commented out
    assert onbox.cu_device_intersect_box(0,0,0, 0.,0.,1., 0.02, 0.02, 0.02) == (-0.01, 0.01)
    assert onbox.cu_device_intersect_box(0,0,0, 1.,1.,0., 0.02, 0.02, 0.02) == (-0.01, 0.01)
    assert onbox.cu_device_intersect_box(0,0,0, 1.,1.,1., 0.02, 0.02, 0.02) == (-0.01, 0.01)
    assert onbox.cu_device_intersect_box(0,0,0, 0.,0.,1., 0.02, 0.02, 0.04) == (-0.02, 0.02)
    return

# guvectorize right now not supported by CUDASIM
@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_cu_intersect_box():
    N = int(1e7)
    N = int(100)
    x_np = np.arange(-0.02, 0.02, 0.04/N, dtype='float32')
    y_np = np.arange(-0.02, 0.02, 0.04/N, dtype='float32')
    z_np = np.ones(x_np.size, dtype='float32')*(-.05)
    vx_np = np.zeros(x_np.size, dtype='float32')
    vy_np = np.zeros(x_np.size, dtype='float32')
    vz_np = np.ones(x_np.size, dtype='float32')
    ts_dim_np = np.zeros(2, dtype='float32')
    # x_cu = cuda.to_device(x_np)
    # y_cu = cuda.to_device(y_np)
    # z_cu = cuda.to_device(z_np)
    # vx_cu = cuda.to_device(vx_np)
    # vy_cu = cuda.to_device(vy_np)
    # vz_cu = cuda.to_device(vz_np)
    # ts_dim_cu = cuda.to_device(ts_dim_np)
    print("starting")
    t1 = time.time()
    sx = sy = sz = 0.05
    # ts_cu = onbox.cu_intersect_box(
    ts = onbox.cu_intersect_box(
        # x_cu, y_cu, z_cu, vx_cu, vy_cu, vz_cu,
        x_np, y_np, z_np, vx_np, vy_np, vz_np,
        sx, sy, sz,
        # ts_dim_cu,
        ts_dim_np
    )
    t2 = time.time()
    print("done calculating")
    print(t2-t1)
    # ts = ts_cu.copy_to_host()
    print(ts[:10])
    return

def main():
    test_cu_device_update_intersections()
    test_cu_device_intersect_box()
    test_cu_intersect_box()
    return

if __name__ == '__main__': main()

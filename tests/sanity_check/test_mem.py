#!/usr/bin/env python
"""
This example shows how to write both the CUDA and the CPU code and test them.
"""

import numpy as np
import math
import numba
from mcvine.acc.test import USE_CUDA
if USE_CUDA:
    from numba import cuda
    import cupy as cp

def cu_square(x, x2):
    N, = x.shape
    for i in range(N):
        x2[i] = x[i]**2
    if USE_CUDA:
        return x2
    else:
        return
if USE_CUDA:
    cu_square = numba.guvectorize(["float32[:], float32[:]"], '(N) -> (N)', target='cuda')(cu_square)
else:
    cu_square = numba.guvectorize(["float32[:], float32[:]"], '(N) -> (N)')(cu_square)

import pytest
@pytest.mark.skipif(not USE_CUDA, reason='No CUDA')
def test_cuda():
    N = int(1e6)
    x_np = np.arange(-0.02, 0.02, 0.04/N, dtype='float32')
    print('transfer')
    x_dev = cuda.to_device(x_np)
    print('square')
    x2 = cu_square(x_dev)
    print(x_np)
    print(x2)
    return

def test_cpu():
    N = int(1e6)
    x_np = np.arange(-0.02, 0.02, 0.04/N, dtype='float32')
    x2 = cu_square(x_np)
    return

def main():
    if USE_CUDA:
        test_cuda()
    else:
        test_cpu()

if __name__ == '__main__': main()

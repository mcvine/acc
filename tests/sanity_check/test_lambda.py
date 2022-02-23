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

    @cuda.jit(device=True)
    def atomic_func(a, x):
        f = lambda x: math.sin(a*x)
        return f(x)

    @cuda.jit
    def kernel(out):
        ind = cuda.grid(1)
        out[ind] = atomic_func(ind/5, math.pi/2)

import pytest
@pytest.mark.skipif(not USE_CUDA, reason='No CUDA')
def test_cuda():
    out = np.zeros(20, dtype='float32')
    kernel[4, 5](out)
    print(out)
    return

def main():
    if USE_CUDA:
        test_cuda()

if __name__ == '__main__': main()

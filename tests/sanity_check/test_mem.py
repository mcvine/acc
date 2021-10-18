#!/usr/bin/env python

from numba import cuda
import numpy as np
import math
import numba
import cupy as cp

@numba.guvectorize(["float32[:], float32[:]"], '(N) -> (N)', target='cuda')
def cu_square(x, x2):
    N, = x.shape
    for i in range(N):
        x2[i] = x[i]**2
    return x2

def main():
    N = int(1e6)
    x_np = np.arange(-0.02, 0.02, 0.04/N, dtype='float32')
    print('transfer')
    x_dev = cuda.to_device(x_np)
    print('square')
    x2 = cu_square(x_dev)
    print(x_np)
    print(x2)
    return

if __name__ == '__main__': main()

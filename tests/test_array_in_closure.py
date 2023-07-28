#!/usr/bin/env python
# copied from
# https://github.com/numba/numba/issues/9084
from numba import cuda
import numpy as np

N = 100000
# closed_array = np.ones(N)
closed_array = cuda.to_device(np.ones(N))

@cuda.jit(cache=False)
def kernel(r, x):
    r[0] = closed_array[x]

r = np.zeros(1)

kernel[1, 1](r, 2)
print(r[0])


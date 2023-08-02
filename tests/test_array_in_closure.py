#!/usr/bin/env python
import pytest
from numba import cuda
import numpy as np

from mcvine.acc import test


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_array_in_closure():
    # Simple test to check that Numba doesn't make a const copy of a global array
    # copied from
    # https://github.com/numba/numba/issues/9084
    # NOTE: this test will currently fail on Numba versions without the 
    #  patch from the linked issue above. 
    #  This patch is included in this repo at: .github/workflows/numba-0.56.4.patch

    N = 100000
    # closed_array = np.ones(N)
    closed_array = cuda.to_device(np.ones(N))

    @cuda.jit(cache=False)
    def kernel(r, x):
        r[0] = closed_array[x]

    r = np.zeros(1)

    kernel[1, 1](r, 2)
    print(r[0])

    assert r[0] == 1.0


if __name__ == '__main__':
    pytest.main([__file__])

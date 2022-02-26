#!/usr/bin/env python

import os, pytest, time
thisdir = os.path.dirname(__file__)
from mcvine.acc import test
import math, numba as nb
from numba import cuda

from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states
from mcni import neutron_buffer, neutron
from mcvine.acc.components.sources import source_simple
from mcvine.acc.components.optics import guide
from mcvine.acc.config import get_numba_floattype
FLOAT = get_numba_floattype()

src = source_simple.Source_simple(
    'src',
    radius = 0., width = 0.03, height = 0.03, dist = 1.,
    xw = 0.035, yh = 0.035,
    Lambda0 = 10., dLambda = 9.5, E0=0., dE=0.0,
    flux=1, gauss=False, N=1
)
guide1 = guide.Guide(
    'guide',
    w1=0.035, h1=0.035, w2=0.035, h2=0.035, l=10,
    R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003,
)

def call_process_no_buffer(
        N,
        # source
        square, width, height, radius,
        wl_distr, Lambda0, dLambda, E0, dE,
        xw, yh, dist, pmul,
        # guide
        ww, hh, hw1, hh1, l,
        R0, Qc, alpha, m, W,
):
    neutron_count = N
    threads_per_block = 512
    nblocks = math.ceil(neutron_count / threads_per_block)
    print("{} blocks, {} threads".format(nblocks, threads_per_block))
    rng_states = create_xoroshiro128p_states(threads_per_block * nblocks, seed=1)
    process_kernel_no_buffer[nblocks, threads_per_block](
        N,
        square, width, height, radius,
        wl_distr, Lambda0, dLambda, E0, dE,
        xw, yh, dist, pmul,
        ww, hh, hw1, hh1, l,
        R0, Qc, alpha, m, W,
        rng_states,
    )
    cuda.synchronize()

@cuda.jit
def process_kernel_no_buffer(
        N,
        square, width, height, radius,
        wl_distr, Lambda0, dLambda, E0, dE,
        xw, yh, dist, pmul,
        ww, hh, hw1, hh1, l,
        R0, Qc, alpha, m, W,
        rng_states
):
    x = cuda.grid(1)
    if x < N:
        neutron = cuda.local.array(shape=10, dtype=FLOAT)
        source_simple.propagate(
            x, rng_states,
            neutron,
            square, width, height, radius,
            wl_distr, Lambda0, dLambda, E0, dE,
            xw, yh, dist, pmul
        )
        neutron[2] -= dist
        guide.propagate(
            neutron,
            ww, hh, hw1, hh1, l,
            R0, Qc, alpha, m, W,
        )
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_no_buffer(N=10):
    t1 = time.time()
    call_process_no_buffer(N, *src.propagate_params, *guide1.propagate_params)
    print(f"Time: {time.time()-t1}")
    return

def main():
    # test_component_no_buffer(N=5)
    test_component_no_buffer(N=1e8)
    return

if __name__ == '__main__': main()

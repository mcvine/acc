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
from mcvine.acc.components.monitors import divpos_monitor
FLOAT = nb.float64

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
mon = divpos_monitor.DivPos_monitor(
    'mon',
    xwidth=0.08, yheight=0.08,
    maxdiv=2.,
    npos=250, ndiv=251
)

def call_process_no_buffer(N, src, guide, mon, out_N, out_p, out_p2, ntotthreads=int(1e5)):
    ntotthreads = min(N, ntotthreads)
    neutron_count = N
    threads_per_block = 512
    nblocks = math.ceil(ntotthreads / threads_per_block)
    print("{} blocks, {} threads".format(nblocks, threads_per_block))
    actual_nthreads = threads_per_block * nblocks
    n_neutrons_per_thread = math.ceil(N / actual_nthreads)
    rng_states = create_xoroshiro128p_states(actual_nthreads, seed=1)
    process_kernel_no_buffer[nblocks, threads_per_block](
        N, n_neutrons_per_thread, src, guide, mon, rng_states, out_N, out_p, out_p2)
    cuda.synchronize()

@cuda.jit
def process_kernel_no_buffer(N, n_neutrons_per_thread, src, guide1, mon, rng_states, out_N, out_p, out_p2):
    dist = 1.
    guide_len = 10.
    gap = 1.
    x = cuda.grid(1)
    start_index = x*n_neutrons_per_thread
    neutron = cuda.local.array(shape=10, dtype=FLOAT)
    for i in range(n_neutrons_per_thread):
        nindex = start_index+i
        if nindex < N:
            r1 = xoroshiro128p_uniform_float32(rng_states, x)
            r2 = xoroshiro128p_uniform_float32(rng_states, x)
            r3 = xoroshiro128p_uniform_float32(rng_states, x)
            r4 = xoroshiro128p_uniform_float32(rng_states, x)
            r5 = xoroshiro128p_uniform_float32(rng_states, x)
            source_simple.propagate(neutron, r1, r2, r3, r4, r5, *src)
            neutron[2] -= dist
            guide.propagate(*guide1, neutron)
            neutron[2] -= guide_len + gap
            divpos_monitor.propagate(neutron, *mon, out_N, out_p, out_p2)
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_no_buffer(N=10, ntotthreads=int(1e5)):
    t1 = time.time()
    call_process_no_buffer(
        N, src._params, guide1._params, mon._params,
        mon.out_N, mon.out_p, mon.out_p2,
        ntotthreads=ntotthreads)
    print(f"Time: {time.time()-t1}")
    return

def main():
    N = 5
    N = 1e11
    test_component_no_buffer(N=N, ntotthreads=int(1e6))
    from matplotlib import pyplot as plt
    plt.pcolormesh(mon.x_centers, mon.div_centers, mon.out_p/N)
    plt.colorbar()
    plt.clim(0, 1e-6)
    plt.show()
    return

if __name__ == '__main__': main()

#!/usr/bin/env python

import os, pytest, time
thisdir = os.path.dirname(__file__)
from mcvine.acc import test
import math, numba as nb, numpy as np
from numba import cuda

from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states
from mcni import neutron_buffer, neutron
from mcvine.acc.components.sources import source_simple
from mcvine.acc.components.optics import guide
from mcvine.acc.components.monitors import divpos_monitor
FLOAT = nb.float64

src1 = source_simple.Source_simple(
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
mon1 = divpos_monitor.DivPos_monitor(
    'mon',
    xwidth=0.08, yheight=0.08,
    maxdiv=2.,
    npos=250, ndiv=250
)


def call_process_no_buffer(N, src, guide, mon, ntotthreads=int(1e5)):
    ntotthreads = min(N, ntotthreads)
    threads_per_block = 512
    nblocks = math.ceil(ntotthreads / threads_per_block)
    actual_nthreads = threads_per_block * nblocks
    n_neutrons_per_thread = math.ceil(N / actual_nthreads)
    print("{} blocks, {} threads, {} neutrons per thread".format(
        nblocks, threads_per_block, n_neutrons_per_thread))
    rng_states = create_xoroshiro128p_states(actual_nthreads, seed=1)
    counter = np.zeros(1, dtype=int)
    process_kernel_no_buffer[nblocks, threads_per_block](
        counter, N, n_neutrons_per_thread, src, guide, mon, rng_states)
    cuda.synchronize()
    print(f"processed {counter.sum():g} neutrons")


source_propagate = source_simple.Source_simple.propagate
guide_propagate = guide.Guide.propagate
monitor_propagate = divpos_monitor.DivPos_monitor.propagate


@cuda.jit
def process_kernel_no_buffer(counter, N, n_neutrons_per_thread, src, guide1, mon, rng_states):
    dist = 1.
    guide_len = 10.
    gap = 1.
    x = cuda.grid(1)
    start_index = x*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    neutron = cuda.local.array(shape=10, dtype=FLOAT)
    for i in range(start_index, end_index):
        source_propagate(x, rng_states, neutron, *src)
        neutron[2] -= dist
        guide_propagate(neutron, *guide1)
        neutron[2] -= guide_len + gap
        monitor_propagate(neutron, *mon)
    cuda.atomic.add(counter, 0, max(end_index-start_index, 0))
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_no_buffer(N=10, ntotthreads=int(1e5)):
    t1 = time.time()
    call_process_no_buffer(
        N, src1.propagate_params, guide1.propagate_params, mon1.propagate_params,
        ntotthreads=ntotthreads)
    print(f"Time: {time.time()-t1}")
    return

def main():
    N = 5
    N = 1e8
    test_component_no_buffer(N=N, ntotthreads=int(1e6))
    from matplotlib import pyplot as plt
    plt.pcolormesh(mon1.x_centers, mon1.div_centers, mon1.out_p/N)
    plt.colorbar()
    plt.clim(0, 1e-6)
    plt.show()
    return

if __name__ == '__main__': main()

import os
thisdir = os.path.dirname(__file__)
import math, numpy as np
from numba import cuda
import pytest
from mcvine.acc import test

@pytest.fixture
def composite_example_interact_path1():
    path = os.path.join(thisdir, "sampleassemblies", 'sample+2cylinders', 'sampleassembly.xml')
    from mcvine.acc.components.samples import loadScattererComposite
    composite = loadScattererComposite(path)
    from mcvine.acc.scatterers.composite_scatterer import factory
    methods = factory(composite)
    return methods['interact_path1']

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_cudasim(composite_example_interact_path1):
    neutron = np.array([0.0,0,0, 0,0,1000, 0,0, 0, 1.])
    threadindex = 0
    rng_states = None
    N = 10 #000
    p = 0.
    absorbed = 0
    for i in range(N):
        neutron = np.array([0.0,0,0, 0,0,1000, 0,0, 0, 1.])
        composite_example_interact_path1(threadindex, rng_states, neutron)
        print(neutron)
        p1 = neutron[-1]
        if p1>0:
            p+=p1
        else:
            absorbed += 1
    p/=N
    absorbed/=N
    print(p, absorbed)
    assert p < 0.1
    assert absorbed == 0
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='no CUDA')
def test_cuda(composite_example_interact_path1):
    @cuda.jit
    def interact_path1_kernel(rng_states, neutrons, n_neutrons_per_thread):
        N = len(neutrons)
        thread_index = cuda.grid(1)
        start_index = thread_index*n_neutrons_per_thread
        end_index = min(start_index+n_neutrons_per_thread, N)
        for i in range(start_index, end_index):
            composite_example_interact_path1(thread_index, rng_states, neutrons[i])
    def run(neutrons):
        N = len(neutrons)
        threads_per_block = 128
        ntotthreads = int(1e5)
        nblocks = math.ceil(ntotthreads / threads_per_block)
        actual_nthreads = threads_per_block * nblocks
        n_neutrons_per_thread = math.ceil(N / actual_nthreads)
        from numba.cuda.random import create_xoroshiro128p_states
        from mcvine.acc.config import rng_seed
        rng_states = create_xoroshiro128p_states(actual_nthreads, seed=rng_seed)
        interact_path1_kernel[nblocks, threads_per_block](rng_states, neutrons, n_neutrons_per_thread)
    N = int(10)
    neutrons = np.zeros((N, 10))
    neutron = np.array([0.0,0,0, 0,0,1000, 0,0, 0, 1.])
    neutrons[:] = neutron
    run(neutrons)
    print(neutrons)
    return

import os
thisdir = os.path.dirname(__file__)
import math, numpy as np
from numba import cuda
import pytest
from mcvine.acc import test

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_homogeneous_scatterer():
    path = os.path.join(thisdir, "sampleassemblies", 'isotropic_sphere', 'sampleassembly.xml')
    from mcvine.acc.components.samples import loadFirstHomogeneousScatterer
    hs = loadFirstHomogeneousScatterer(path)
    shape = hs.shape()
    kernel = hs.kernel()
    mcweights = 1., 1., 1.
    packing_factor = 0.6
    from mcvine.acc.scatterers.homogeneous_scatterer import factory
    methods = factory(shape, kernel, mcweights, packing_factor)
    interact_path1 = methods['interact_path1']
    threadindex = 0
    rng_states = None
    N = 10000
    p = 0.
    absorbed = 0
    for i in range(N):
        neutron = np.array([0.0,0,0, 0,0,1000, 0,0, 0, 1.])
        interact_path1(threadindex, rng_states, neutron)
        p1 = neutron[-1]
        if p1>0:
            p+=p1
        else:
            absorbed += 1
    p/=N
    absorbed/=N
    assert p<.8 and p>.7
    assert absorbed < 0.36 and absorbed > 0.3
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_scatter_func_factory():
    path = os.path.join(thisdir, "sampleassemblies", 'isotropic_sphere', 'sampleassembly.xml')
    from mcvine.acc.components.samples import loadFirstHomogeneousScatterer
    hs = loadFirstHomogeneousScatterer(path)
    from mcvine.acc.scatterers import scatter_func_factory
    methods = scatter_func_factory.render(hs)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='no CUDA')
def test_homogeneous_scatterer_cuda():
    path = os.path.join(thisdir, "sampleassemblies", 'isotropic_sphere', 'sampleassembly.xml')
    from mcvine.acc.components.samples import loadFirstHomogeneousScatterer
    hs = loadFirstHomogeneousScatterer(path)
    shape = hs.shape()
    kernel = hs.kernel()
    mcweights = 1., 1., 1.
    packing_factor = 0.6
    from mcvine.acc.scatterers.homogeneous_scatterer import factory
    methods = factory(shape, kernel, mcweights, packing_factor)
    interact_path1 = methods['interact_path1']
    @cuda.jit
    def interact_path1_kernel(rng_states, neutrons, n_neutrons_per_thread):
        N = len(neutrons)
        thread_index = cuda.grid(1)
        start_index = thread_index*n_neutrons_per_thread
        end_index = min(start_index+n_neutrons_per_thread, N)
        for i in range(start_index, end_index):
            interact_path1(thread_index, rng_states, neutrons[i])
    def run(neutrons):
        N = len(neutrons)
        threads_per_block = 512
        ntotthreads = int(1e5)
        nblocks = math.ceil(ntotthreads / threads_per_block)
        actual_nthreads = threads_per_block * nblocks
        n_neutrons_per_thread = math.ceil(N / actual_nthreads)
        from numba.cuda.random import create_xoroshiro128p_states
        from mcvine.acc.config import rng_seed
        rng_states = create_xoroshiro128p_states(actual_nthreads, seed=rng_seed)
        interact_path1_kernel[nblocks, threads_per_block](rng_states, neutrons, n_neutrons_per_thread)
    N = 100
    neutrons = np.zeros((N, 10))
    neutron = np.array([0.0,0,0, 0,0,1000, 0,0, 0, 1.])
    neutrons[:] = neutron
    run(neutrons)
    return


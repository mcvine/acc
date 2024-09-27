import numpy as np
import pytest
from mcni import neutron_buffer, neutron
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
from mcni.utils import conversion
from mcvine.acc import test
from mcvine.acc.config import rng_seed
from mcvine.acc.kernels import Phonon_IncoherentElastic
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states


# Simple wrapper kernel to test the scatter device function
@cuda.jit()
def scatter_test_kernel(
        rng_states, N, n_neutrons_per_thread, neutrons, dw_core
):
    thread_index = cuda.grid(1)
    start_index = thread_index * n_neutrons_per_thread
    end_index = min(start_index + n_neutrons_per_thread, N)
    for neutron_index in range(start_index, end_index):
        Phonon_IncoherentElastic.scatter(
            neutron_index, rng_states, neutrons[neutron_index], dw_core)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_IncoherentElastic_kernel():

    tof_at_sample = 1.0
    dw_core = 0.1

    vil = 3000.0
    n = neutron([0., 0., -5.0], [0., 0., vil], prob=1.0, time=tof_at_sample)

    # calculate initial vi and ei
    vi = np.linalg.norm(n.state.velocity)
    Ei = conversion.v2e(vi)

    # create neutron buffer
    ncount = 10
    buffer = neutron_buffer(ncount)
    for i in range(ncount):
        buffer[i] = n
    tmp = neutrons_as_npyarr(buffer)
    tmp.shape = -1, ndblsperneutron
    buffer_d = cuda.to_device(tmp)

    # setup test kernel with 1 neutron
    nblocks = ncount
    threads_per_block = 1
    rng_states = create_xoroshiro128p_states(
        nblocks * threads_per_block, seed=rng_seed)

    scatter_test_kernel[nblocks, threads_per_block](
        rng_states, ncount, 1, buffer_d, dw_core)

    cuda.synchronize()
    buffer = buffer_d.copy_to_host()

    # calculate Q and E and compare against expected
    print(buffer)


if __name__ == '__main__':
    pytest.main([__file__])

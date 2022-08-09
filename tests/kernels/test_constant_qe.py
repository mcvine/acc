import numpy as np
import pytest
from mcni import neutron_buffer, neutron
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
from mcvine.acc import test
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from acc.config import rng_seed
from acc.kernels import constant_qe


# Simple wrapper kernel to test the scatter device function
@cuda.jit
def scatter_test_kernel(
        rng_states, N, n_neutrons_per_thread, neutrons, Q, E
):
    thread_index = cuda.grid(1)
    start_index = thread_index * n_neutrons_per_thread
    end_index = min(start_index + n_neutrons_per_thread, N)
    for neutron_index in range(start_index, end_index):
        constant_qe.S(neutron_index, rng_states, neutrons[neutron_index], Q, E)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_constant_qe_kernel():
    Q = 3
    E = 30

    Ei = 60.0
    vi = np.sqrt(Ei) * constant_qe.sqrte2v
    n = neutron([0., 0., 0.], [0., 0., vi])

    # create neutron buffer
    buffer = neutron_buffer(1)
    buffer[0] = n
    tmp = neutrons_as_npyarr(buffer)
    tmp.shape = -1, ndblsperneutron
    buffer_d = cuda.to_device(tmp)

    # calculate initial vi and ei
    vi = np.linalg.norm(n.state.velocity)

    # Ei = constant_qe.v2E(vi)
    Ei = vi * vi * constant_qe.vsq2e

    # setup test kernel with 1 neutron
    nblocks = 1
    threads_per_block = 1
    rng_states = create_xoroshiro128p_states(nblocks * threads_per_block, seed=rng_seed)

    scatter_test_kernel[nblocks, threads_per_block](rng_states, 1, 1, buffer_d, Q, E)

    cuda.synchronize()
    buffer = buffer_d.copy_to_host().flatten()

    # calculate Q and E and compare against expected
    vf = np.linalg.norm(buffer[3:6])
    Ef = vf * vf * constant_qe.vsq2e
    e_diff = Ei - Ef
    v_diff = buffer[3:6] - np.array(n.state.velocity)
    q_as_v = np.linalg.norm(v_diff)
    q_actual = constant_qe.v2k * q_as_v

    np.testing.assert_almost_equal(e_diff, E)
    np.testing.assert_almost_equal(q_actual, Q)


if __name__ == '__main__':
    pytest.main([__file__])

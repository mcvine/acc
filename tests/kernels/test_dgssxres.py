import numpy as np
import pytest
from mcni import neutron_buffer, neutron
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
from mcni.utils import conversion
from mcvine.acc import test
from mcvine.acc.config import rng_seed
from mcvine.acc.kernels import DGSSXResKernel
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states


# Simple wrapper kernel to test the scatter device function
@cuda.jit()
def scatter_test_kernel(
        rng_states, N, n_neutrons_per_thread, neutrons, target, target_radius, tof_target, delta_tof
):
    thread_index = cuda.grid(1)
    start_index = thread_index * n_neutrons_per_thread
    end_index = min(start_index + n_neutrons_per_thread, N)
    for neutron_index in range(start_index, end_index):
        DGSSXResKernel.scatter(
            neutron_index, rng_states, neutrons[neutron_index], target, target_radius, tof_target, delta_tof)


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_constant_qe_kernel():

    tof_at_sample = 1.0
    target_position = np.asarray([3.0, 0.0, 0.0], dtype=float)
    target_radius = 0.025
    tof_at_target = 0.001 + tof_at_sample
    dtof = 1e-5

    vil = 3000.0
    n = neutron([0., 0., 0.], [0., 0., vil], prob=1.0, time=tof_at_sample)

    # calculate initial vi and ei
    vi = np.linalg.norm(n.state.velocity)
    Ei = conversion.v2e(vi)

    # create neutron buffer
    buffer = neutron_buffer(1)
    buffer[0] = n
    tmp = neutrons_as_npyarr(buffer)
    tmp.shape = -1, ndblsperneutron
    buffer_d = cuda.to_device(tmp)

    # setup test kernel with 1 neutron
    nblocks = 1
    threads_per_block = 1
    rng_states = create_xoroshiro128p_states(
        nblocks * threads_per_block, seed=rng_seed)

    scatter_test_kernel[nblocks, threads_per_block](
        rng_states, 1, 1, buffer_d, target_position, target_radius, tof_at_target, dtof)

    cuda.synchronize()
    buffer = buffer_d.copy_to_host().flatten()

    # calculate Q and E and compare against expected
    vf = np.linalg.norm(buffer[3:6])
    Ef = conversion.v2e(vf)
    e_diff = Ei - Ef
    v_diff = buffer[3:6] - np.array(n.state.velocity)
    q_as_v = np.linalg.norm(v_diff)
    q_actual = conversion.V2K * q_as_v

    np.testing.assert_allclose(e_diff, 0.0, atol=0.5)
    np.testing.assert_allclose(q_actual, 6.74, rtol=1.0e-2)


if __name__ == '__main__':
    pytest.main([__file__])

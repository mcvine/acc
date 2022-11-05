#!/usr/bin/env python

import os, pytest, numpy as np
thisdir = os.path.abspath(os.path.dirname(__file__))
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from mcvine.acc import test

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_interactM1():
    from HMS_isotropic_sphere import HMS
    _interactM1 = HMS._interactM1
    @cuda.jit
    def run_kernel(rng_states, neutron, out_neutrons):
        thread_id = cuda.grid(1)
        _interactM1(thread_id, rng_states, out_neutrons, neutron)
        return
    neutron = np.array([0,0,-1, 0,0,3000., 0,0, 0., 1.])
    out_neutrons = np.zeros((2, 10), dtype=float)
    rng_states = create_xoroshiro128p_states(1, seed=0)
    run_kernel[1,1](rng_states, neutron, out_neutrons)
    print(out_neutrons)
    return

def main():
    test_interactM1()
    return

if __name__ == '__main__': main()

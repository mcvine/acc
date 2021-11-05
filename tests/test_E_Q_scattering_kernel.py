#!/usr/bin/env python

import pytest, os
if os.environ.get('USE_CUDA').lower() == 'false':
    pytest.skip("No CUDA", allow_module_level=True)

import mcvine.acc.E_Q_scattering_kernel as eqs
import numpy as np

def test_E_Q_kernel():
    neutron_velocity = np.array([[0.,0.,8000.]]*10000)
    neutron_probability = [1.]*10000
    Qmin =1.
    Qmax =5.
    E_Q = '10*Q'
    S_Q = 1.
    scattering_coefficient = 1.0
    absorption_cross_section = 1.0

    scattered_neutron_probability , scattered_neutron_velocity = eqs.E_Q_scattering_kernel_call(
        Qmin, Qmax,
        neutron_velocity,
        neutron_probability,
        E_Q, S_Q,
        scattering_coefficient,
        absorption_cross_section)
    assert(len(scattered_neutron_probability) == 10000)
    assert (scattered_neutron_velocity.shape[0] == 10000)
    assert (scattered_neutron_velocity.shape[1] == 3)
    return

def main():
    test_E_Q_kernel()
    return

if __name__ == '__main__': main()

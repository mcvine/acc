#!/usr/bin/env python

import pytest
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from mcni import neutron_buffer, neutron
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
from mcni.utils import conversion
from mcvine.acc import test
from mcvine.acc.config import rng_seed
from mcvine.acc.kernels import E_Q as E_Q_kernel

def test_makeS():
    E_Q = '5*sin(Q)**2+5'
    S_Q = '1'
    S = E_Q_kernel.makeS(E_Q, S_Q, 0, 15)
    in_neutron = np.array([0.,0.,0., 0.,0.,2000., 0.,0., 0., 1.])
    for i in range(10):
        neutron = in_neutron.copy()
        S(0, None, neutron)
        vi = in_neutron[3:6]
        vf = neutron[3:6]
        ki = conversion.V2K*vi
        kf = conversion.V2K*vf
        Q = ki - kf
        Q = np.linalg.norm(Q)
        Ei = conversion.v2e(np.linalg.norm(vi))
        Ef = conversion.v2e(np.linalg.norm(vf))
        E = Ei - Ef
        assert np.isclose(E, eval(E_Q, dict(Q=Q, sin=np.sin)))
    return

def main():
    test_makeS()
    return

if __name__ == '__main__': main()

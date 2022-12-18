#!/usr/bin/env python

import os, pytest
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from mcni import neutron_buffer, neutron
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
from mcni.utils import conversion
from mcvine.acc import test
from mcvine.acc.config import rng_seed
from mcvine.acc.kernels import E_Q as E_Q_kernel

thisdir = os.path.dirname(__file__)
path = os.path.join(thisdir, "UN", 'sampleassembly.xml')
from mcvine.acc.components.samples import loadFirstHomogeneousScatterer
hs = loadFirstHomogeneousScatterer(path)
shape = hs.shape()
kernel = hs.kernel()

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_makeKernelModule():
    from mcvine.acc.kernels.composite import makeKernelModule
    makeKernelModule(kernel)
    return

def main():
    test_makeKernelModule()
    return

if __name__ == '__main__': main()

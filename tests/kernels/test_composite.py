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

def test_makeKernelModule():
    from mcvine.acc.kernels.composite import makeKernelModule
    makeKernelModule(kernel)
    return

def test_render():
    from mcvine.acc.kernels import scatter_func_factory
    scatter, calc_scattering_coeff, absorb = scatter_func_factory.render(kernel)
    return

def main():
    test_makeKernelModule()
    test_render()
    return

if __name__ == '__main__': main()

#!/usr/bin/env python

import os, pytest
thisdir = os.path.dirname(__file__)
from mcvine.acc import test

from mcni import neutron_buffer, neutron
from mcvine.acc.components.sources.source_simple import Source_simple
src = Source_simple(
    'src',
    radius=0.05, height=0, width=0, dist=10.0,
    xw=0.1, yh=0.1,
    E0=60, dE=10, Lambda0=0, dLambda=0,
    flux=1, gauss=False, N=1
)

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component():
    neutrons = src.process(neutron_buffer(10))
    for n in neutrons:
        print(n)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_long(ncount = 1e6):
    neutrons = src.process(neutron_buffer(int(ncount)))
    return

def test_mcstas_component_long(ncount=1e6):
    import mcvine.components as mc
    src = mc.sources.Source_simple(
        'src', S_filename=dat,
        Emin=5, Emax=20,
        width = 0.03, height=0.03,
        dist = 5., xw=0.03, yh=0.03
    )
    neutrons = src.process(neutron_buffer(int(ncount)))
    return

def main():
    # test_component()
    test_component_long(1e7)
    # test_mcstas_component_long(1e7)
    return

if __name__ == '__main__': main()

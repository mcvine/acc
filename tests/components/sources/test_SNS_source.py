#!/usr/bin/env python

import os, pytest
thisdir = os.path.dirname(__file__)
dat = os.path.join(thisdir, 'source_rot2_cdr_cyl_3x3_20190417.dat')
from mcvine.acc import test

from mcni import neutron_buffer, neutron
from mcvine.acc.components.sources.SNS_source import SNS_source

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component():
    src = SNS_source('src', dat, 5, 20, 0.03, 0.03, 5, .03, .03)
    neutrons = src.process(neutron_buffer(10))
    for n in neutrons:
        print(n)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_long(ncount = 1e6):
    src = SNS_source('src', dat, 5, 20, 0.03, 0.03, 5, .03, .03)
    neutrons = src.process(neutron_buffer(int(ncount)))
    return

def test_mcstas_component_long(ncount=1e6):
    import mcvine.components as mc
    src = mc.sources.SNS_source(
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

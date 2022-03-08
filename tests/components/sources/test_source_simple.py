#!/usr/bin/env python

import os, pytest
thisdir = os.path.dirname(__file__)
from mcvine.acc import test

from mcni import neutron_buffer, neutron
import mcvine.components as mc
from mcvine import run_script

thisdir = os.path.dirname(__file__)

from mcvine.acc.components.sources.source_simple import Source_simple
def src_factory(name='src'):
    return Source_simple(
        name,
        radius=0., height=0.03, width=0.03, dist=5.0,
        xw=0.03, yh=0.03,
        E0=0, dE=0, Lambda0=10., dLambda=9.5,
        flux=1, gauss=False, N=1
    )
src = src_factory()

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_no_buffer(N=10):
    src.process_no_buffer(N)
    return

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

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_long_with_monitors(ncount = 1e6):
    instr = os.path.join(thisdir, "src_psdmon.py")
    outdir = 'out.debug-acc_source_simple'
    ncount = int(ncount)
    run_script.run1(
        instr, outdir,
        ncount=ncount, buffer_size=ncount,
        source_factory=src_factory,
        overwrite_datafiles=True)
    return

def main():
    # test_component_no_buffer(N=5)
    # test_component_no_buffer(N=1e8)
    # test_component()
    # test_component_long(1e7)
    test_component_long_with_monitors(1e7)
    # test_mcstas_component_long(1e7)
    return

if __name__ == '__main__': main()

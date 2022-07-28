#!/usr/bin/env python

import os, pytest
thisdir = os.path.dirname(__file__)
from mcvine.acc import test

from mcni import neutron_buffer, neutron
import mcvine.components as mc
from mcvine import run_script
from mcvine.acc import run_script as acc_run_script

thisdir = os.path.abspath(os.path.dirname(__file__))

from mcvine.acc.components.sources.neutronfromstorage import NeutronFromStorage
def create_src_factory(filename):
    def src_factory(name='src'):
        path = os.path.join(thisdir, filename)
        print(path)
        return NeutronFromStorage(name, path)
    return src_factory
src = create_src_factory("singletestneutron.mcv")()

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
    outdir = 'out.debug-acc_neutronfromstorage'
    ncount = int(ncount)
    run_script.run1(
        instr, outdir,
        ncount=ncount, buffer_size=ncount,
        source_factory=create_src_factory("singletestneutron2.mcv"),
        overwrite_datafiles=True)
    outdir = 'out.debug-acc_neutronfromstorage-fullacc'
    acc_run_script.run(
        instr, outdir,
        ncount=ncount,
        source_factory=create_src_factory("singletestneutron2.mcv"),
        all_acc_components=True,
        overwrite_datafiles=True)
    return

def main():
    # test_component_no_buffer(N=5)
    # test_component_no_buffer(N=1e8)
    # test_component()
    # test_component_long(1e7)
    test_component_long_with_monitors(10)
    # test_component_long_with_monitors(1e7)
    return

if __name__ == '__main__': main()

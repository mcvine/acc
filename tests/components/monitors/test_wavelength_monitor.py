#!/usr/bin/env python

import os, pytest
thisdir = os.path.dirname(__file__)
from mcvine.acc import test

from mcni import neutron_buffer, neutron
import mcvine.components as mc
from mcvine.acc import run_script

thisdir = os.path.dirname(__file__)

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_long(ncount = 1e6):
    instr = os.path.join(thisdir, "src_mon_instrument.py")
    outdir = 'out.wavelength_monitor'
    ncount = int(ncount)
    run_script.run(instr, outdir, ncount=ncount)
    return

def main():
    test_component_long(1e7)
    return

if __name__ == '__main__': main()

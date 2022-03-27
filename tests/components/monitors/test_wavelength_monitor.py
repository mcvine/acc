#!/usr/bin/env python

import os, pytest
thisdir = os.path.dirname(__file__)
from mcvine.acc import test

from mcni import neutron_buffer, neutron
import mcvine.components as mc
from mcvine.acc import run_script

thisdir = os.path.dirname(__file__)

def wavelength_monitor():
    from mcvine.acc.components.monitors.wavelength_monitor import Wavelength_monitor
    return Wavelength_monitor(
        "mon",
        xwidth=0.03, yheight=0.03,
        Lmin=0, Lmax=10., nchan=200,
        filename = "IL.h5"
    )

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_long(ncount = 1e6):
    instr = os.path.join(thisdir, "src_mon_instrument.py")
    outdir = 'out.wavelength_monitor'
    ncount = int(ncount)
    run_script.run(instr, outdir, ncount=ncount, monitor_factory=wavelength_monitor)
    return

def main():
    test_component_long(1e7)
    return

if __name__ == '__main__': main()

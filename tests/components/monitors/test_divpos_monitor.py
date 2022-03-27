#!/usr/bin/env python

interactive = False

import os, pytest
thisdir = os.path.dirname(__file__)
from mcvine.acc import test

from mcni import neutron_buffer, neutron
import mcvine.components as mc
from mcvine.acc import run_script

thisdir = os.path.dirname(__file__)

def divpos_monitor():
    from mcvine.acc.components.monitors.divpos_monitor import DivPos_monitor
    return DivPos_monitor(
        "mon",
        xwidth=0.08, yheight=0.08,
        maxdiv=1.,
        npos=100, ndiv=250,
        filename = "divpos.h5",
    )

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_long(ncount = 1e6):
    instr = os.path.join(thisdir, "src_mon_instrument.py")
    outdir = 'out.divpos_monitor'
    ncount = int(ncount)
    run_script.run(instr, outdir, ncount=ncount, monitor_factory=divpos_monitor)
    import histogram.hdf as hh
    hist = hh.load(os.path.join(outdir, "divpos.h5"))
    from histogram import plot as plotHist
    plotHist(hist)
    return

def main():
    global interactive
    interactive = True
    test_component_long(1e7)
    return

if __name__ == '__main__': main()

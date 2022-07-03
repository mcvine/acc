#!/usr/bin/env python

interactive = False

import os, pytest
thisdir = os.path.dirname(__file__)
from mcvine.acc import test

from mcni import neutron_buffer, neutron
import mcvine.components as mc
from mcvine.acc import run_script

thisdir = os.path.dirname(__file__)

def psd_monitor_4pi():
    from mcvine.acc.components.monitors.psd_monitor_4pi import PSD_monitor_4Pi
    return PSD_monitor_4Pi(
        "mon",
        nphi=100, ntheta=120,
        filename = "psd_4pi.h5",
    )

def source():
    from mcvine.acc.components.sources.source_simple import Source_simple
    return Source_simple(
        "source",
        radius=0., height=0.01, width=0.01, dist=1.0,
        xw=2, yh=2,
        E0=0, dE=0, Lambda0=5., dLambda=2,
        flux=1, gauss=False, N=1
    )

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_component_long(ncount = 1e6):
    instr = os.path.join(thisdir, "src_mon_instrument.py")
    outdir = 'out.psd_monitor_4pi'
    ncount = int(ncount)
    run_script.run(
        instr, outdir, ncount=ncount,
        source_factory = source,
        monitor_factory=psd_monitor_4pi, monitor_z=0)
    if interactive:
        import histogram.hdf as hh
        hist = hh.load(os.path.join(outdir, "psd_4pi.h5"))
        from histogram import plot as plotHist
        plotHist(hist)
    return

def main():
    global interactive
    interactive = True
    test_component_long(1e7)
    return

if __name__ == '__main__': main()

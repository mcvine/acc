#!/usr/bin/env python

import os, pytest, numpy as np
thisdir = os.path.abspath(os.path.dirname(__file__))
from mcvine.acc import test

script = os.path.join(thisdir, 'acc_ms_test_instrument.py')
workdir = 'out.acc_ms'
ncount = int(1e5)

from mcvine.acc import run_script

def halo_ms_sample():
    from HMS_halo_isotropic_sphere import HMS
    return HMS('sample')

def psd_monitor_4pi():
    from mcvine.acc.components.monitors.psd_monitor_4pi import PSD_monitor_4Pi
    return PSD_monitor_4Pi(
        "mon",
        nphi=190, ntheta=190, radius=3,
        filename = "psd_4pi.h5",
    )

def psd_mon_factory():
    from mcvine.acc.components.monitors.psd_monitor import PSD_monitor
    return PSD_monitor(
        name='mon', nx=500, ny=500,
        xwidth=1.0,
        yheight=1.0
    )

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compile():
    run_script.compile(
        script, sample_factory=halo_ms_sample, monitor_factory=psd_monitor_4pi)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_run():
    #run_script.run(script, workdir, ncount=ncount, monitor_factory=psd_monitor_4pi)
    run_script.run(
        script, workdir, ncount=ncount,
        sample_factory=halo_ms_sample,
        monitor_factory=psd_mon_factory
    )
    return


def main():
    test_compile()
    test_run()

    #monitor_hist = os.path.join(workdir, "psd_4pi.h5")
    monitor_hist = os.path.join(workdir, "psd.h5")
    import histogram.hdf as hh
    from histogram import plot as plotHist
    plotHist(hh.load(monitor_hist))

    return

if __name__ == '__main__': main()

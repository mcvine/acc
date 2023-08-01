#!/usr/bin/env python

import os, shutil
import pytest
from mcvine.acc import test
from mcvine import run_script

thisdir = os.path.dirname(__file__)

def test_gpu(ncount=1e6):
    Ei = 70.0
    mod2sample = 10.0
    R = 3.0
    instrument = os.path.join(thisdir, 'acc_ss_test_instrument.py')
    workdir = 'out.test_iqe_mon'
    def source():
        from mcvine.acc.components.sources.source_simple import Source_simple
        return Source_simple(
            'src',
            radius = 0., width = 0.01, height = 0.01, dist = mod2sample-0.5,
            xw = 0.008, yh = 0.008,
            E0 = Ei, dE=0.1, Lambda0=0, dLambda=0.,
            flux=1, gauss=0
        )
    def sample():
        from HSS_fccAl_E_Q_kernel_box import HSS
        return HSS(name='sample')
    def monitor():
        from mcvine.acc.components.monitors.dgs_iqe_monitor import IQE_monitor
        return IQE_monitor(
            'iqe_monitor',
            Ei = Ei, L0=mod2sample,
            Qmin=0., Qmax=8.0, nQ = 160,
            Emin=-60.0, Emax=60.0, nE = 120,
            min_angle_in_plane=-45., max_angle_in_plane=135.,
            min_angle_out_of_plane=-45., max_angle_out_of_plane=45.,
            radius = R, filename = "iqe.h5"
        )
    from mcvine.acc import run_script
    run_script.run(
        instrument, workdir, ncount,
        source_factory=source, sample_factory=sample, monitor_factory=monitor,
        z_sample = mod2sample,
    )
    return

def main():
    import journal
    journal.info("instrument").activate()
    test_gpu()
    return


if __name__ == '__main__': main()

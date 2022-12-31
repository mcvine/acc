#!/usr/bin/env python

import os, shutil
import pytest
from mcvine.acc import test
from mcvine import run_script

thisdir = os.path.dirname(__file__)

def test_cpu():
    instr = os.path.join(thisdir, "fccAl_E_Q_box_instrument.py")
    outdir = 'out.fccAl_E_Q_box'
    if os.path.exists(outdir): shutil.rmtree(outdir)
    ncount = 1e5
    run_script.run1(
        instr, outdir,
        ncount=ncount, buffer_size=int(ncount),
        is_acc=False,
    )
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_gpu(ncount=1e6):
    instrument = os.path.join(thisdir, 'acc_ss_test_instrument.py')
    workdir = 'out.fccAl_E_Q_box-allacc'
    def source():
        from mcvine.acc.components.sources.source_simple import Source_simple
        return Source_simple(
            'src',
            radius = 0., width = 0.01, height = 0.01, dist = 1.,
            xw = 0.008, yh = 0.008,
            E0 = 70.0, dE=0.1, Lambda0=0, dLambda=0.,
            flux=1, gauss=0
        )
    def sample():
        from HSS_fccAl_E_Q_kernel_box import HSS
        return HSS(name='sample')
    def monitor():
        from mcvine.acc.components.monitors.iqe_monitor import IQE_monitor
        return IQE_monitor(
            'iqe_monitor',
            Ei = 70.0,
            Qmin=0., Qmax=8.0, nQ = 160,
            Emin=-60.0, Emax=60.0, nE = 120,
            min_angle_in_plane=0., max_angle_in_plane=359.,
            min_angle_out_of_plane=-90., max_angle_out_of_plane=90.,
        )
    from mcvine.acc import run_script
    run_script.run(
        instrument, workdir, ncount,
        source_factory=source, sample_factory=sample, monitor_factory=monitor,
    )
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e6), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    instr = os.path.join(thisdir, "fccAl_E_Q_box_instrument.py")
    relerr_tol = dict(threshold=0.05, outlier_fraction=0.02)
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "fccAl_E_Q_box",
        ["IQE"],
        {"float32": 4e-12, "float64": 4e-12},
        num_neutrons, debug,
        instr=instr,
        interactive=interactive,
        acc_component_spec = dict(is_acc=True),
        nonacc_component_spec = dict(is_acc=False),
        relerr_tolerances = dict(float32=relerr_tol, float64=relerr_tol)
    )


def main():
    import journal
    journal.info("instrument").activate()
    # test_cpu()
    test_gpu()
    # test_compare_mcvine(num_neutrons=int(100), interactive=True, debug=True)
    # test_compare_mcvine(num_neutrons=int(1e6), interactive=True)
    return


if __name__ == '__main__': main()

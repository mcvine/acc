#!/usr/bin/env python

import os, shutil
import pytest
from mcvine.acc import test

thisdir = os.path.dirname(__file__)

def ms_sample():
    from HMS_UN_block import HMS
    return HMS('sample')

def iqe_monitor():
    from mcvine.acc.components.monitors.psd_monitor_4pi import PSD_monitor_4Pi
    return PSD_monitor_4Pi(
        "mon",
        nphi=190, ntheta=190, radius=3,
        filename = "psd_4pi.h5",
    )

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_gpu(ncount = 1e5):
    script = os.path.join(thisdir, 'acc_ms_test_instrument.py')
    workdir = 'out.ms_UN_block-gpu'
    from mcvine.acc import run_script
    run_script.run(
        script, workdir, ncount=ncount,
        sample_factory=ms_sample,
        monitor_factory=iqe_monitor,
    )
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e6), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    instr = os.path.join(thisdir, "ms_UN_block_instrument.py")
    relerr_tol = dict(threshold=0.10, outlier_fraction=0.10)
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "ms_UN_block",
        ["IQE"],
        {"float32": 2e-12, "float64": 2e-12},
        num_neutrons, debug,
        instr=instr,
        interactive=interactive,
        acc_component_spec = dict(is_acc=True),
        nonacc_component_spec = dict(is_acc=False, multiple_scattering=True),
        relerr_tolerances = dict(float32=relerr_tol, float64=relerr_tol)
    )
    if interactive:
        compareI_Q()

def compareI_Q():
    import numpy as np, histogram.hdf as hh
    gpu = hh.load('./out.debug-ms_un_block_gpu_instrument/IQE.h5')
    cpu = hh.load('./out.debug-mcvine_ms_un_block_cpu_instrument/IQE.h5')
    Es = np.arange(0., 360., 50.)
    gpu_I_Q, cpu_I_Q = [], []
    for E in Es:
        gpu_I_Q.append( gpu[(), (E-10, E+10)].sum('energy') )
        cpu_I_Q.append( cpu[(), (E-10, E+10)].sum('energy') )
    from matplotlib import pyplot as plt
    plt.figure()
    for E, gs, cs in zip(Es, gpu_I_Q, cpu_I_Q):
        plt.plot(gs.Q, gs.I, 'r--', label=f"GPU: {E}")
        plt.plot(cs.Q, cs.I, 'k', label=f"CPU: {E}")
    plt.show()
    return

def main():
    import journal
    journal.info("instrument").activate()
    # test_compare_mcvine(num_neutrons=int(1e5), interactive=True)
    test_gpu()
    return


if __name__ == '__main__': main()

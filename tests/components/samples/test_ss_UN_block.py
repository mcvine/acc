#!/usr/bin/env python

import os, shutil
import pytest
from mcvine.acc import test
from mcvine import run_script

thisdir = os.path.dirname(__file__)

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e6), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    instr = os.path.join(thisdir, "ss_UN_block_instrument.py")
    relerr_tol = dict(threshold=0.10, outlier_fraction=0.10)
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "ss_UN_block",
        ["IQE"],
        {"float32": 2e-12, "float64": 2e-12},
        num_neutrons, debug,
        instr=instr,
        interactive=interactive,
        acc_component_spec = dict(is_acc=True),
        nonacc_component_spec = dict(is_acc=False),
        relerr_tolerances = dict(float32=relerr_tol, float64=relerr_tol)
    )
    if interactive:
        compareI_Q()

def compareI_Q():
    import numpy as np, histogram.hdf as hh
    gpu = hh.load('./out.debug-ss_un_block_gpu_instrument/IQE.h5')
    cpu = hh.load('./out.debug-mcvine_ss_un_block_cpu_instrument/IQE.h5')
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
    test_compare_mcvine(num_neutrons=int(1e7), interactive=True)
    return


if __name__ == '__main__': main()

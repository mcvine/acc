#!/usr/bin/env python

import os
import shutil
import pytest
import numpy as np
from mcvine.acc import test
from mcvine import run_script

thisdir = os.path.dirname(__file__)


def test_cpu():
    instr = os.path.join(thisdir, "fccNi_PhononIncohEl_sphere_instrument.py")
    outdir = 'out.fccNi_PhononIncohEl_sphere'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    ncount = 1e5
    run_script.run1(
        instr, outdir,
        ncount=ncount, buffer_size=int(ncount),
        is_acc=False,
    )
    return


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1024), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    instr = os.path.join(thisdir, "fccNi_PhononIncohEl_sphere_instrument.py")
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "fccNi_PhononIncohEl_sphere",
        ["IQE"],
        {"float32": 4e-10, "float64": 4e-10},
        num_neutrons, debug,
        instr=instr,
        interactive=interactive,
        acc_component_spec=dict(is_acc=True),
        nonacc_component_spec=dict(is_acc=False),
    )

    gpu_file = os.path.join("./out.debug-fccni_phononincohel_sphere_gpu_instrument", "IQE.h5")
    cpu_file = os.path.join("./out.debug-mcvine_fccni_phononincohel_sphere_cpu_instrument", "IQE.h5")

    import histogram.hdf as hh
    gpu_hist = hh.load(gpu_file)
    cpu_hist = hh.load(cpu_file)

    # integrate the energies
    gpu_E_hist = gpu_hist.sum("energy")
    cpu_E_hist = cpu_hist.sum("energy")

    # subtract the integrated CPU and GPU intensities
    diff_hist = gpu_E_hist - cpu_E_hist

    if interactive:    
        from histogram import plot as plotHist
        plotHist(gpu_hist)
        plotHist(cpu_hist)

        plotHist(gpu_E_hist)
        plotHist(cpu_E_hist)
        plotHist(diff_hist)

    assert np.mean(diff_hist.I) < 1.0e-12


def main():
    import journal
    journal.info("instrument").activate()
    # test_cpu()
    # test_compare_mcvine(num_neutrons=int(100), interactive=True, debug=True)
    test_compare_mcvine(num_neutrons=int(1e6), interactive=True)
    return


if __name__ == '__main__':
    main()

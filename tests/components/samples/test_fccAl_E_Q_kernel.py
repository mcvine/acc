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
    # test_compare_mcvine(num_neutrons=int(100), interactive=True, debug=True)
    test_compare_mcvine(num_neutrons=int(1e6), interactive=True)
    return


if __name__ == '__main__': main()

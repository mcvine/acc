#!/usr/bin/env python

import os, shutil
import pytest
from mcvine.acc import test
from mcvine import run_script

thisdir = os.path.dirname(__file__)


def test1():
    instr = os.path.join(thisdir, "fccAl_diffr_sphere_instrument.py")
    outdir = 'out.fccAl_diffr_sphere'
    if os.path.exists(outdir): shutil.rmtree(outdir)
    ncount = 1e5
    run_script.run1(
        instr, outdir,
        ncount=ncount, buffer_size=int(ncount),
    )
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e7), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    instr = os.path.join(thisdir, "fccAl_diffr_sphere_instrument.py")
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "fccAl_diffr_sphere",
        ["psd_4pi"],
        {"float32": 4e-10, "float64": 4e-10},
        num_neutrons, debug,
        instr = instr,
        interactive=interactive,
        acc_component_spec = dict(is_acc=True),
        nonacc_component_spec = dict(is_acc=False),
    )

def main():
    import journal
    journal.info("instrument").activate()
    # test1()
    # test_compare_mcvine(num_neutrons=int(100), interactive=True, debug=True)
    test_compare_mcvine(num_neutrons=int(1e7), interactive=True)
    return


if __name__ == '__main__': main()

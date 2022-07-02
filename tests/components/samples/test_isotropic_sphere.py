#!/usr/bin/env python

import os, shutil
import pytest
from mcvine.acc import test
from mcvine import run_script

thisdir = os.path.dirname(__file__)


def test1():
    instr = os.path.join(thisdir, "isotropic_sphere_instrument.py")
    outdir = 'out.isotropic_sphere'
    if os.path.exists(outdir): shutil.rmtree(outdir)
    run_script.run1(
        instr, outdir,
        ncount=1e5, buffer_size=int(1e5),
    )
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e7), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    instr = os.path.join(thisdir, "isotropic_sphere_instrument.py")
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "isotropic_sphere",
        ["psd_4pi"],
        {"float32": 2e-10, "float64": 2e-10},
        num_neutrons, debug,
        instr = instr,
        interactive=interactive,
        acc_component_spec = dict(is_acc=True),
        nonacc_component_spec = dict(is_acc=False),
    )


def main():
    test_compare_mcvine(num_neutrons=int(1e7), interactive=True)
    # test_compare_mcvine(num_neutrons=int(100), interactive=True, debug=True)
    # test1()
    return


if __name__ == '__main__': main()

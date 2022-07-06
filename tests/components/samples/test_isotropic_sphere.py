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
    instr = os.path.join(thisdir, "isotropic_sphere_instrument.py")
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "isotropic_sphere",
        ["psd_4pi"],
        {"float32": 4e-10, "float64": 4e-10},
        num_neutrons, debug,
        instr = instr,
        interactive=interactive,
        acc_component_spec = dict(is_acc=True),
        nonacc_component_spec = dict(is_acc=False),
    )

def psd_monitor_4pi():
    from mcvine.acc.components.monitors.psd_monitor_4pi import PSD_monitor_4Pi
    return PSD_monitor_4Pi(
        "mon",
        nphi=30, ntheta=30, radius=3,
        filename = "psd_4pi.h5",
    )

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_acc_run_script(ncount = 1e6):
    instr = os.path.join(thisdir, "acc_ss_instrument.py")
    outdir = 'out.isotropic_sphere-acc_run_script'
    ncount = int(ncount)
    from mcvine.acc import run_script
    run_script.run(
        instr, outdir, ncount=ncount,
        monitor_factory=psd_monitor_4pi,
        z_sample = 1.,
    )
    return

def main():
    import journal
    journal.info("instrument").activate()
    # test1()
    # test_compare_mcvine(num_neutrons=int(100), interactive=True, debug=True)
    # test_compare_mcvine(num_neutrons=int(1e7), interactive=True)
    test_acc_run_script(ncount=1e7)
    return


if __name__ == '__main__': main()

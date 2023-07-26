#!/usr/bin/env python

import os, shutil
import pytest
from mcvine.acc import test
from mcvine import run_script

thisdir = os.path.dirname(__file__)


def test_1():
    instr = os.path.join(thisdir, "test_instrument_with_samplefromxml.py")
    samplexml = "sampleassemblies/isotropic_sphere/sampleassembly.xml"
    outdir = 'out.test_instrument_source_composite_monitor_isotropic_sphere'
    if os.path.exists(outdir): shutil.rmtree(outdir)
    ncount = 1e5
    run_script.run1(
        instr, outdir,
        ncount=ncount,
        samplexml = samplexml,
        factory = "mcvine.acc.components.samples.composite.sampleassembly_from_xml",
    )
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e7), debug=False, interactive=False):
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    instr = os.path.join(thisdir, "test_instrument_with_samplefromxml.py")
    samplexml = "sampleassemblies/isotropic_sphere/sampleassembly.xml",
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "isotropic_sphere",
        ["psd_4pi"],
        {"float32": 4e-10, "float64": 4e-10},
        num_neutrons, debug,
        instr = instr,
        interactive=interactive,
        acc_component_spec = dict(
            samplexml=samplexml,
            factory = "mcvine.acc.components.samples.composite.sampleassembly_from_xml",
        ),
        nonacc_component_spec = dict(
            samplexml=samplexml,
            factory="mcvine.components.samples.SampleAssemblyFromXml",
        ),
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
    # instr = os.path.join(thisdir, "acc_ss_instrument.py")
    instr = os.path.join(thisdir, "acc_composite_isotropic_sphere_instrument.py")
    outdir = 'out.composite_isotropic_sphere-acc_run_script'
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
    # test_1()
    # test_compare_mcvine(num_neutrons=int(100), interactive=True, debug=True)
    # test_compare_mcvine(num_neutrons=int(1e7), interactive=True)
    test_acc_run_script(ncount=1e6)
    return


if __name__ == '__main__': main()

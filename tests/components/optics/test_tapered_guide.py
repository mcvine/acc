#!/usr/bin/env python

import os
import histogram.hdf as hh
import numpy as np
import pytest
import shutil
from mcni import neutron_buffer, neutron
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
from mcvine import run_script
from mcvine.acc import test
from mcvine.acc.components.optics.guide_tapering import Guide


thisdir = os.path.dirname(__file__)
interactive = False


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine():
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    num_neutrons = int(1e6)
    # Run the mcvine instrument first
    instr = os.path.join(thisdir, "tapered_guide_instrument.py")
    cpu_outdir = 'out.debug-tapered_guide_cpu_instrument'
    if os.path.exists(cpu_outdir):
        shutil.rmtree(cpu_outdir)
    run_script.run1(
        instr, cpu_outdir, ncount=num_neutrons,
        guide_factory = "mcvine.components.optics.Guide_tapering",
        overwrite_datafiles=True)

    outdir = 'out.debug-tapered_guide_gpu_instrument'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    run_script.run1(
        instr, outdir, ncount=num_neutrons,
        guide_mod = "mcvine.acc.components.optics.guide_tapering",
        overwrite_datafiles=True)

    # Compare output files
    cpu_Ixy = hh.load(os.path.join(cpu_outdir, "Ixy.h5"))
    cpu_Ixdivx = hh.load(os.path.join(cpu_outdir, "Ixdivx.h5"))
    Ixy = hh.load(os.path.join(outdir, "Ixy.h5"))
    Ixdivx = hh.load(os.path.join(outdir, "Ixdivx.h5"))

    global interactive
    if interactive:
        from histogram import plot as plotHist
        plotHist(cpu_Ixy)
        plotHist(cpu_Ixdivx)
        plotHist(Ixy)
        plotHist(Ixdivx)
    assert cpu_Ixy.shape() == Ixy.shape()
    assert cpu_Ixdivx.shape() == Ixdivx.shape()
    assert np.allclose(cpu_Ixy.I, Ixy.I)
    assert np.allclose(cpu_Ixdivx.I, Ixdivx.I)
    return


def main():
    global interactive
    interactive = True
    test_compare_mcvine()
    return


if __name__ == '__main__':
    main()

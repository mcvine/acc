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
from mcvine.acc.config import get_numpy_floattype

thisdir = os.path.dirname(__file__)
interactive = False


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e7), debug=False):
    """
    Tests the acc cpu implementation of a straight guide against mcvine
    """
    if debug:
        assert num_neutrons < 1001
    # Run the mcvine instrument first
    instr = os.path.join(thisdir, "guide_instrument.py")
    mcvine_outdir = 'out.debug-mcvine_guide_cpu_instrument'
    if os.path.exists(mcvine_outdir):
        shutil.rmtree(mcvine_outdir)
    run_script.run1(
        instr, mcvine_outdir,
        ncount=num_neutrons, buffer_size=num_neutrons,
        guide_factory="mcvine.components.optics.Guide",
        save_neutrons_after_guide=debug,
        overwrite_datafiles=True)

    # Run our guide implementation
    outdir = 'out.debug-guide_gpu_instrument'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    run_script.run1(
        instr, outdir,
        ncount=num_neutrons, buffer_size=num_neutrons,
        guide_mod="mcvine.acc.components.optics.guide",
        save_neutrons_after_guide=debug,
        overwrite_datafiles=True, )

    # Compare output files
    mcvine_Ixy = hh.load(os.path.join(mcvine_outdir, "Ixy.h5"))
    mcvine_Ixdivx = hh.load(os.path.join(mcvine_outdir, "Ixdivx.h5"))
    Ixy = hh.load(os.path.join(outdir, "Ixy.h5"))
    Ixdivx = hh.load(os.path.join(outdir, "Ixdivx.h5"))

    global interactive
    if interactive:
        from histogram import plot as plotHist
        plotHist(mcvine_Ixy)
        plotHist(mcvine_Ixdivx)
        plotHist(Ixy)
        plotHist(Ixdivx)
        plotHist((Ixy-mcvine_Ixy)/mcvine_Ixy)
    assert mcvine_Ixy.shape() == Ixy.shape()
    assert mcvine_Ixdivx.shape() == Ixdivx.shape()

    tolerance = 1e-7 if get_numpy_floattype() == np.float32 else 1e-8
    assert np.allclose(mcvine_Ixy.data().storage(), Ixy.data().storage(),
                       atol=tolerance)
    assert np.allclose(mcvine_Ixdivx.data().storage(), Ixdivx.data().storage(),
                       atol=tolerance)


def debug():
    global interactive
    interactive = True
    test_compare_mcvine(debug=True, num_neutrons=100)
    return


def main():
    global interactive
    interactive = True
    test_compare_mcvine(num_neutrons=int(1e6))
    return


if __name__ == '__main__':
    main()
    # debug()

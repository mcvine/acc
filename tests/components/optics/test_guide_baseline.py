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
from mcvine.acc.components.optics.guide import Guide
from mcvine.acc.geometry.plane import Plane


thisdir = os.path.dirname(__file__)
interactive = False


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons = int(1e6), debug=False):
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
        guide_factory = "mcvine.components.optics.Guide",
        save_neutrons_after_guide=debug,
        overwrite_datafiles=True)

    # Run our guide implementation
    outdir = 'out.debug-guide_gpu_instrument'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    run_script.run1(
        instr, outdir,
        ncount=num_neutrons, buffer_size=num_neutrons,
        guide_mod = "mcvine.acc.components.optics.guide_baseline",
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
    assert histogram_is_close(
        Ixy.I, mcvine_Ixy.I,
        min_nonzero_fraction=0.92,
        rtol=1e-3, min_rdiff_fraction=0.98,
        atol=1e-7*1e-3, min_adiff_fraction=0.99
    )
    assert histogram_is_close(
        Ixdivx.I, mcvine_Ixdivx.I,
        min_nonzero_fraction=0.2,
        rtol=1e-3, min_rdiff_fraction=0.99,
        atol=1e-7*1e-3, min_adiff_fraction=0.99
    )
    return

def histogram_is_close(
        h1, h2, min_nonzero_fraction=0.9,
        rtol=0.001, min_rdiff_fraction=0.9,
        atol=1e-10, min_adiff_fraction=0.9,
):
    "check if histogram h1 is close to the reference h2"
    if h2.shape != h1.shape: return False
    rdiff = h1/h2-1
    isfinite = np.isfinite(rdiff)
    rdiff = rdiff[isfinite]
    adiff = (h1-h2)[~isfinite]
    # make sure there is enough data points that are not zero
    print(rdiff.size, h2.size)
    if rdiff.size<min_nonzero_fraction*h2.size: return False
    # make sure for a good portion of nonzero data points,
    # the relative diff is smaller than the given tolerance
    print ((np.abs(rdiff)<rtol).sum() , rdiff.size)
    if (np.abs(rdiff)<rtol).sum() < rdiff.size*min_rdiff_fraction:
        return False
    print ((np.abs(adiff)<atol).sum() , adiff.size)
    if (np.abs(adiff)<atol).sum() < adiff.size*min_adiff_fraction:
        return False
    return True

def debug():
    global interactive
    interactive = True
    test_compare_mcvine(debug=True, num_neutrons=100)
    return

def main():
    global interactive
    interactive = True
    test_compare_mcvine()
    return


if __name__ == '__main__':
    main()
    # debug()

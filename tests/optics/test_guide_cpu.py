#!/usr/bin/env python

import os, shutil
import histogram.hdf as hh
thisdir = os.path.dirname(__file__)
from mcvine import run_script
interactive = False

def test():
    '''
    Tests the acc cpu implementation of a straight guide against mcvine
    '''
    num_neutrons = 100000
    # Run the mcvine instrument first
    mcvine_instr = os.path.join(thisdir, "mcvine_guide_cpu_instrument.py")
    mcvine_outdir = 'out.debug-mcvine_guide_cpu_instrument'
    if os.path.exists(mcvine_outdir): shutil.rmtree(mcvine_outdir)
    run_script.run1(mcvine_instr, mcvine_outdir, ncount=num_neutrons, overwrite_datafiles=True)

    # Run our guide implementation
    instr = os.path.join(thisdir, "guide_cpu_instrument.py")
    outdir = 'out.debug-guide_cpu_instrument'
    if os.path.exists(outdir): shutil.rmtree(outdir)
    run_script.run1(instr, outdir, ncount=num_neutrons, overwrite_datafiles=True)

    # TODO: Compare output files
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
    return

def main():
    global interactive
    interactive = True
    test()
    return

if __name__ == '__main__': main()

#!/usr/bin/env python

import os, shutil
thisdir = os.path.dirname(__file__)
from mcvine import run_script
interactive = False

def test():
    instr = os.path.join(thisdir, "guide_cpu_instrument.py")
    outdir = 'out.debug-guide_cpu_instrument'
    if os.path.exists(outdir): shutil.rmtree(outdir)
    run_script.run1(instr, outdir, ncount=100000, overwrite_datafiles=True)
    global interactive
    if interactive:
        from histogram import plot as plotHist
        import histogram.hdf as hh
        Ixy = hh.load(os.path.join(outdir, "Ixy.h5"))
        plotHist(Ixy)
        Ixdivx = hh.load(os.path.join(outdir, "Ixdivx.h5"))
        plotHist(Ixdivx)
    return

def main():
    global interactive
    interactive = True
    test()
    return

if __name__ == '__main__': main()

#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)
from mcvine import run_script

def test():
    instr = os.path.join(thisdir, "guide_cpu_instrument.py")
    outdir = 'out.debug-guide_cpu_instrument'
    run_script.run1(instr, outdir, ncount=100000, overwrite_datafiles=True)
    return

def main():
    test()
    return

if __name__ == '__main__': main()

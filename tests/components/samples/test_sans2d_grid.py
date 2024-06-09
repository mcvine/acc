#!/usr/bin/env python

import os, shutil
import pytest
from mcvine.acc import test
from mcvine.acc import run_script

thisdir = os.path.dirname(__file__)

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test1():
    import make_sans2d_data
    make_sans2d_data.make_sans2d_data()
    instr = os.path.join(thisdir, "sans2d_grid_instrument.py")
    outdir = 'out.sans2d_grid'
    if os.path.exists(outdir): shutil.rmtree(outdir)
    ncount = 2e6
    run_script.run(
        instr, outdir,
        ncount=ncount, buffer_size=int(ncount),
    )
    return

def main():
    test1()
    return

if __name__ == '__main__': main()

#!/usr/bin/env python

import os, shutil
thisdir = os.path.dirname(__file__)
from mcvine import run_script

from mcvine import run_script
instr = os.path.join(thisdir, "test_instrument.py")
ncount = 1e10
outdir = f'out.nonacc_src_guide_mon_n{ncount}'
if os.path.exists(outdir): shutil.rmtree(outdir)
ncount = int(ncount)
run_script.run_mpi(
# run_script.run1(
    instr, outdir,
    nodes=40,
    ncount=ncount, buffer_size=1e6,
    overwrite_datafiles=True)

#!/usr/bin/env python

import os, pytest
thisdir = os.path.abspath(os.path.dirname(__file__))
from mcvine.acc import test

script = os.path.join(thisdir, 'VERDI_base.py')
workdir = 'out.verdi_cpu'
ncount = int(1e6)
nodes = 2

from mcvine import run_script

def test_run1():
    run_script.run1(
        script, workdir=workdir, ncount=ncount, buffer_size=int(1e6),
        use_gpu=False, overwrite_datafiles=True)
    return

def test_run_mpi():
    run_script.run_mpi(
        script, workdir=workdir,
        ncount=ncount, buffer_size=int(5e5), nodes=nodes,
        use_gpu=False, overwrite_datafiles=True)
    return

def main():
    test_run1()
    # test_run_mpi()
    return

if __name__ == '__main__': main()

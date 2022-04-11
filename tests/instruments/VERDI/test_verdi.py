#!/usr/bin/env python

import os, pytest
thisdir = os.path.abspath(os.path.dirname(__file__))
from mcvine.acc import test

script = os.path.join(thisdir, 'VERDI_base.py')
workdir = 'out.verdi'
ncount = int(1e7)

from mcvine.acc import run_script

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compile():
    run_script.compile(script)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_run():
    run_script.run(
        script, workdir=workdir, ncount=ncount,
        ntotalthreads=int(1e6/512*256), threads_per_block=256)
    return

def main():
    test_run()
    # test_compile()
    return

if __name__ == '__main__': main()

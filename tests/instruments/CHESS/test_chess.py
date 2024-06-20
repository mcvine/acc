#!/usr/bin/env python

import os, pytest
thisdir = os.path.abspath(os.path.dirname(__file__))
from mcvine.acc import test

script = os.path.join(thisdir, 'simple.py')
workdir = 'out.simple'

from mcvine.acc import run_script

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compile():
    run_script.compile(script)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_run(ncount=1e4):
    run_script.run(
        script, workdir=workdir, ncount=ncount,
        ntotalthreads=min(ncount, int(1e6)),
        threads_per_block=32,
    )
    return

def main():
    # test_compile()
    test_run(int(1e6))
    return

if __name__ == '__main__': main()

#!/usr/bin/env python

import os, pytest
thisdir = os.path.abspath(os.path.dirname(__file__))
from mcvine.acc import test

script = os.path.join(thisdir, 'acc_sgm_instrument.py')
workdir = 'out.acc_sgm'
ncount = int(1e8)

from mcvine.acc import run_script

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compile():
    run_script.compile(script)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_run():
    run_script.run(script, workdir, ncount=ncount)
    return


def main():
    test_run()
    return

if __name__ == '__main__': main()

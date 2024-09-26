#!/usr/bin/env python

import os, pytest
thisdir = os.path.abspath(os.path.dirname(__file__))
from mcvine.acc import test
from mcvine.acc.test.utils import check_histogram_match

script = os.path.join(thisdir, 'acc_sgm_instrument.py')
workdir = 'out.acc_sgm'
ncount = int(1e8)

from mcvine.acc import run_script

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compile():
    run_script.compile(script)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compile_buffered():
    run_script.compile_buffered(script)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_run():
    run_script.run(script, workdir, ncount=ncount)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_run_buffered():
    # run with ~1e6 neutrons, 2 neutrons per thread - for this test, this needs 2 neutrons per
    # thread to ensure the same RNG states for each neutron when run with two iterations below
    run_script.run(script, workdir, ncount=int(2 ** 20),
                   ntotalthreads=int(2 ** 19), use_buffer=True)
    # run again with a smaller buffer size to force multiple iterations
    # ~1e6 neutrons, 1 neutron per thread, but 2 iterations of 1024 blocks to work within the max buffer size
    workdir_small_buffer = os.path.join(workdir, 'small_buffer')
    run_script.run(script, workdir_small_buffer, ncount=int(2 ** 20),
                   ntotalthreads=int(2 ** 20), use_buffer=True,
                   buffer_size=int(2 ** 19))

    # ensure both results match
    assert check_histogram_match(os.path.join(workdir, "posdiv.h5"),
                                 os.path.join(workdir_small_buffer, "posdiv.h5"),
                                 interactive=True)


def main():
    test_run()
    os.system("plothist --min=0 --max=1e-6 {}/posdiv.h5".format(workdir))
    return

if __name__ == '__main__': main()

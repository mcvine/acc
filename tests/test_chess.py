#!/usr/bin/env python

import os, pytest
thisdir = os.path.abspath(os.path.dirname(__file__))
from mcvine.acc import test, run_script

gpu_script = os.path.join(thisdir, 'chess_test_instrument.py')


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compile():
    run_script.compile(gpu_script)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_run_gpu(ncount=int(1e6), interactive=False):
    workdir = 'out.chess_test'

    # All GPU components
    run_script.run(gpu_script, workdir=workdir, ncount=ncount)

    if interactive:
        plot_chess(workdir, "L_Guide")
        plot_chess(workdir, "L_Sample")
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_run_cpu(ncount=int(1e6), interactive=False):
    from mcvine.run_script import run1
    
    # Test with all CPU components except for MultiDiskChopper component
    cpu_script = os.path.join(thisdir, 'chess_test_instrument_cpu.py')
    workdir = 'out.cpu.chess_test'

    run1(cpu_script, workdir=workdir,
         ncount=ncount, buffer_size=ncount,
         overwrite_datafiles=True)

    if interactive:
        plot_chess(workdir, "L_Guide")
        plot_chess(workdir, "L_Sample")
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_run_cpu2(ncount=int(1e6), interactive=False):
    from mcvine.run_script import run1
    
    # All GPU components except for CPU L_Monitor
    cpu_script = os.path.join(thisdir, 'chess_test_instrument_cpu2.py')
    workdir = 'out.cpu.chess_test2'

    run1(cpu_script, workdir=workdir,
         ncount=ncount, buffer_size=ncount,
         overwrite_datafiles=True)

    if interactive:
        plot_chess(workdir, "L_Guide")
        plot_chess(workdir, "L_Sample")
    return

def plot_chess(workdir, monitor_name):
    import histogram.hdf as hh
    from histogram import plot as plotHist

    fname = os.path.join(workdir, monitor_name + ".h5")
    print("Reading histogram file {}".format(fname))

    hist = hh.load(fname)
    print(hist.axes)
    plotHist(hist)

def main():
    #test_run_cpu(int(1e7), True)
    test_run_cpu2(int(1e7), True)
    test_run_gpu(int(1e7), True)
    return

if __name__ == '__main__': main()

#!/usr/bin/env python

"""
Performance comparison
@quary:
* CPU (Intel Core i7-4790K 4GHz) 1e7 10cores
  - real    2m22.877s
  - user    16m20.316s
  - sys     0m33.205s
* GPU (GeForce GTX 1660) 1e9
  - real    7m32.941s
  - user    7m19.961s
  - sys     0m13.396s
"""

import os, sys, shutil
import pytest
from mcvine.acc import test

thisdir = os.path.abspath(os.path.dirname(__file__))
if thisdir not in sys.path: sys.path.insert(0, thisdir)
import UN_test_instrument as uti, plot_UN_IQ
from test_ss_UN import compareIQs, plotIQcomparison

script = os.path.join(thisdir, 'UN_test_instrument.py')
Ei = 500.0
cpu_workdir = 'out.ms_UN-cpu'
gpu_workdir = 'out.ms_UN-gpu'

def run_cpu(ncount = 1e6, interactive=False):
    workdir = cpu_workdir
    from mcvine import run_script
    logfile = 'log.ms_UN-cpu'
    try:
        run_script.run_mpi(
            script, workdir, overwrite_datafiles=True,
            multiple_scattering=True,
            ncount=ncount, nodes=10, buffer_size=5e4,
            Ei = Ei, log=logfile,
        )
    except RuntimeError as e:
        msg = "CPU simulation failed. Log file: {os.path.abspath(logfile)}\n"
        msg += str(e)
        raise RuntimeError(msg)
    if interactive:
        plot_UN_IQ.plot(os.path.join(workdir, 'iqe.h5'))
    return

def run_gpu(ncount = 1e6, interactive=False):
    workdir = gpu_workdir
    def sample():
        from UN_HMS import HMS
        return HMS(name='sample')
    from mcvine.acc import run_script
    run_script.run(
        script, workdir, ncount=ncount,
        Ei = Ei,
        source_factory = uti.source_gpu,
        sample_factory = sample,
        monitor_factory = uti.monitor_gpu,
    )
    if interactive:
        plot_UN_IQ.plot(os.path.join(workdir, 'iqe.h5'))
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_cpu_vs_gpu(interactive=False):
    run_gpu(ncount=1e6)
    run_cpu(ncount=1e6)
    Es, cpu_I_Q, gpu_I_Q = compareIQs(
        cpu_workdir, gpu_workdir,
        relerr = None, outlier_fraction = None
    )
    if interactive:
        labels = dict(cpu='CPU ncount=1e7', gpu='GPU ncount=1e9')
        plotIQcomparison(Es, cpu_I_Q, gpu_I_Q, labels=labels)
    return

def main():
    import journal
    journal.info("instrument").activate()
    # run_gpu(ncount=1e9)
    # run_cpu(ncount=1e7)
    test_cpu_vs_gpu(interactive=True)
    return

if __name__ == '__main__': main()

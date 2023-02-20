#!/usr/bin/env python

import os, sys, shutil
import pytest
from mcvine.acc import test

thisdir = os.path.abspath(os.path.dirname(__file__))
if thisdir not in sys.path: sys.path.insert(0, thisdir)
import UN_test_instrument as plot_UN_IQ
import UN_with_ARCS_test_instrument as uati
from test_ss_UN import compareIQs, plotIQcomparison

script = os.path.join(thisdir, 'UN_with_ARCS_test_instrument.py')
Ei = 500.0
cpu_workdir = 'out.ms_UN_with_ARCS-cpu'
gpu_workdir = 'out.ms_UN_with_ARCS-gpu'

def run_cpu(ncount = 1e6, interactive=False):
    workdir = cpu_workdir
    from mcvine import run_script
    logfile = 'log.ms_UN_with_ARCS-cpu'
    try:
        run_script.run_mpi(
            script, workdir, overwrite_datafiles=True,
            multiple_scattering=True,
            ncount=ncount, nodes=10, buffer_size=5e4,
            Ei = Ei, log=logfile,
        )
    except RuntimeError as e:
        msg = f"CPU simulation failed. Log file: {os.path.abspath(logfile)}\n"
        msg += str(e)
        raise RuntimeError(msg)
    if interactive:
        plot_UN_IQ.plot(os.path.join(workdir, 'iqe.h5'))
    return

def run_gpu(ncount = 1e7, interactive=False):
    workdir = gpu_workdir
    def sample():
        from UN_HMS import HMS
        return HMS(name='sample')
    from mcvine.acc import run_script
    run_script.run(
        script, workdir, ncount=ncount,
        Ei = Ei,
        source_factory = uati.source_gpu,
        sample_factory = sample,
        monitor_factory = uati.monitor_gpu2,
    )
    if interactive:
        plot_UN_IQ.plot(os.path.join(workdir, 'iqe.h5'))
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_cpu_vs_gpu(interactive=False):
    run_gpu(ncount=1e9)
    run_cpu(ncount=1e7)
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
    run_gpu(ncount=1e7)
    # run_cpu(ncount=1e7)
    # test_cpu_vs_gpu(interactive=True)
    return

if __name__ == '__main__': main()

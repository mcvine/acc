#!/usr/bin/env python

import os, sys, shutil
import pytest
from mcvine.acc import test

thisdir = os.path.dirname(__file__)
if thisdir not in sys.path: sys.path.insert(0, thisdir)
import UN_test_instrument as uti, plot_UN_IQ

Ei = 500.0
cpu_workdir = 'out.ss_UN-cpu'
gpu_workdir = 'out.ss_UN-gpu'

def run_cpu(ncount = 1e5, interactive=False):
    script = os.path.join(thisdir, 'UN_test_instrument.py')
    workdir = cpu_workdir
    from mcvine import run_script
    run_script.run_mpi(
        script, workdir, overwrite_datafiles=True,
        ncount=ncount, nodes=10,
        Ei = Ei
    )
    if interactive:
        plot_UN_IQ.plot(os.path.join(workdir, 'iqe.h5'))
    return

def run_gpu(ncount = 1e5, interactive=False):
    script = os.path.join(thisdir, 'UN_test_instrument.py')
    workdir = gpu_workdir
    def sample():
        from UN_HSS import HSS
        return HSS(name='sample')
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
def test_cpu_vs_gpu(ncount=int(1e6), interactive=False):
    # run_cpu(ncount = ncount)
    # run_gpu(ncount = ncount)
    Es, cpu_I_Q, gpu_I_Q = compareIQs()
    if interactive:
        plotIQcomparison(Es, cpu_I_Q, gpu_I_Q)

def compareIQs():
    import numpy as np, histogram.hdf as hh
    gpu = hh.load(f'{gpu_workdir}/iqe.h5')
    cpu = hh.load(f'{cpu_workdir}/iqe.h5')
    Es = np.arange(0., 360., 50.)
    gpu_IQ_list, cpu_IQ_list = [], []
    for E in Es:
        gpuIQ = gpu[(), (E-10, E+10)].sum('E')
        gpu_IQ_list.append( gpuIQ )
        cpuIQ = cpu[(), (E-10, E+10)].sum('energy')
        cpu_IQ_list.append( cpuIQ )
        print(cpuIQ.I.max())
        # print(cpuIQ.I-gpuIQ.I)
        outliers = np.abs(cpuIQ.I-gpuIQ.I) > cpuIQ.I.max()*0.1
        # print(cpuIQ.Q[outliers])
        # print(cpuIQ.I[outliers])
        # print(gpuIQ.I[outliers])
        assert outliers.sum() < 0.1 * cpuIQ.size()
    return Es, gpu_IQ_list, cpu_IQ_list

def plotIQcomparison(Es, cpu_I_Q, gpu_I_Q):
    from matplotlib import pyplot as plt
    plt.figure()
    for E, gs, cs in zip(Es, gpu_I_Q, cpu_I_Q):
        plt.plot(gs.Q, gs.I, 'r--', label=f"GPU: {E}")
        plt.plot(cs.Q, cs.I, 'k', label=f"CPU: {E}")
    plt.legend()
    plt.show()
    return

def main():
    import journal
    journal.info("instrument").activate()
    test_cpu_vs_gpu(ncount=int(1e7), interactive=True)
    # run_gpu(ncount=1e8, interactive=True)
    # run_cpu(ncount=1e7, interactive=True)
    return


if __name__ == '__main__': main()

#!/usr/bin/env python

import os, pytest, numpy as np
thisdir = os.path.abspath(os.path.dirname(__file__))
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from mcvine.acc import test

from mcvine.acc import run_script

script = os.path.join(thisdir, 'acc_ms_test_instrument.py')
workdir = 'out.acc_ms'
ncount = int(1e7)

def psd_mon_factory():
    from mcvine.acc.components.monitors.psd_monitor import PSD_monitor
    return PSD_monitor(
        name='mon', nx=1000, ny=1000,
        xwidth=0.5,
        yheight=0.5
    )

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_interactM1():
    from HMS_isotropic_hollowcylinder import HMS
    _interactM1 = HMS._interactM1
    @cuda.jit
    def run_kernel(rng_states, out_neutrons, neutron):
        thread_id = cuda.grid(1)
        _interactM1(thread_id, rng_states, out_neutrons, neutron)
        return
    neutron = np.array([-1,0,0, 3000,0,0., 0,0, 0., 1.])
    out_neutrons = np.zeros((2, 10), dtype=float)
    rng_states = create_xoroshiro128p_states(1, seed=0)
    run_kernel[1,1](rng_states, out_neutrons, neutron)
    print(out_neutrons)
    assert np.allclose(
        out_neutrons[0][:-1],
        [-0.01,0,0, 3000,0,0, 0,0, (1-0.01)/3000.])
    scx = out_neutrons[1][0]
    assert scx < -0.01 and scx > -0.02
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_interactM_path1():
    from HMS_isotropic_hollowcylinder import HMS
    interactM_path1 = HMS.interactM_path1
    @cuda.jit
    def run_kernel(rng_states, out_neutrons, N, neutron):
        thread_id = cuda.grid(1)
        N[thread_id] = interactM_path1(thread_id, rng_states, out_neutrons, neutron)
        return
    neutron = np.array([-1,0,0, 3000,0,0., 0,0, 0., 1.])
    out_neutrons = np.zeros((5, 10), dtype=float)
    rng_states = create_xoroshiro128p_states(1, seed=0)
    N = np.zeros(1, dtype=int)
    rt = run_kernel[1,1](rng_states, out_neutrons, N, neutron)
    assert N[0] == HMS.max_ms_loops_path1 + 1
    print(out_neutrons[:N[0]])
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_scatterM():
    from HMS_isotropic_hollowcylinder import HMS
    scatterM = HMS.scatterM
    @cuda.jit
    def run_kernel(rng_states, out_neutrons, N, neutron):
        thread_id = cuda.grid(1)
        N[thread_id] = scatterM(thread_id, rng_states, out_neutrons, neutron)
        return
    neutron = np.array([-1,0,0, 3000,0,0., 0,0, 0., 1.])
    out_neutrons = np.zeros((10, 10), dtype=float)
    rng_states = create_xoroshiro128p_states(1, seed=0)
    N = np.zeros(1, dtype=int)
    rt = run_kernel[1,1](rng_states, out_neutrons, N, neutron)
    print(N)
    print(out_neutrons[:N[0]])
    return

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test_scatterM_cudasim():
    # rng_states = create_xoroshiro128p_states(1, seed=0)
    rng_states = None
    neutron = np.array([-1,0,0, 3000,0,0., 0,0, 0., 1.])
    out_neutrons = np.zeros((40, 10), dtype=float)

    from HMS_isotropic_hollowcylinder import HMS
    scatterM = HMS.scatterM

    N = scatterM(0, rng_states, out_neutrons, neutron)
    print(N)
    print(out_neutrons[:N])
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compare_mcvine(num_neutrons=int(1e7), debug=False, interactive=False):
    instr = os.path.join(thisdir, "isotropic_hollowcylinder_instrument.py")
    from mcvine.acc.test.compare_acc_nonacc import compare_acc_nonacc
    compare_acc_nonacc(
        "isotropic_hollowcylinder",
        ["psd_4pi"],
        {"float32": 4e-10, "float64": 4e-10},
        num_neutrons, debug,
        instr = instr,
        interactive=interactive,
        acc_component_spec = dict(is_acc=True),
        nonacc_component_spec = dict(is_acc=False),
    )

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_compile():
    run_script.compile(script, monitor_factory=psd_mon_factory)
    return

@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_run():
    
    run_script.run(script, workdir, ncount=ncount, monitor_factory=psd_mon_factory, threads_per_block=128)
    #run_script.run(script, workdir, ncount=ncount)
    
    # plot interactively 
    monitor_hist = os.path.join(workdir, "psd.h5")
    import histogram.hdf as hh
    from histogram import plot as plotHist
    plotHist(hh.load(monitor_hist))
    
    return

def main():
    #test_interactM1()
    #test_interactM_path1()
    #test_scatterM()
    # test_scatterM_cudasim()

    test_run()
    #test_compile()
    #test_compare_mcvine(num_neutrons=int(1e5), interactive=True)
    
    return

if __name__ == '__main__': main()

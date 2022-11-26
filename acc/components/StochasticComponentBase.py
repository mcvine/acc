"""
Requirement for a stochastic component
* component class
  - inherit from StochasticComponentBase
  - ctor must create self.propagate_params
* `propagate` method
  - first three arguments: `threadindex`, `rng_states`, `neutron`
  - other args: match comp.propagate_params
"""

from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states

from ..config import rng_seed, get_numba_floattype
NB_FLOAT = get_numba_floattype()
from .ComponentBase import ComponentBase as base
class StochasticComponentBase(base):

    def call_process(
            self, process_kernel, in_neutrons,
            ntotthreads=int(1e6), threads_per_block=512,
    ):
        N = len(in_neutrons)
        ntotthreads = min(N, ntotthreads)
        nblocks = math.ceil(ntotthreads / threads_per_block)
        actual_nthreads = threads_per_block * nblocks
        n_neutrons_per_thread = math.ceil(N / actual_nthreads)
        print("%s blocks, %s threads, %s neutrons per thread" % (
            nblocks, threads_per_block, n_neutrons_per_thread))
        rng_states = create_xoroshiro128p_states(actual_nthreads, seed=rng_seed)
        process_kernel[nblocks, threads_per_block](
            rng_states, in_neutrons, n_neutrons_per_thread, self.propagate_params)
        cuda.synchronize()
        return in_neutrons

    @classmethod
    def register_propagate_method(cls, propagate):
        new_propagate = cls._adjust_propagate_type(propagate)
        if cls.is_multiscattering:
            cls.process_kernel = make_process_ms_kernel(new_propagate, cls.NUM_MULTIPLE_SCATTER)
        else:
            cls.process_kernel = make_process_kernel(new_propagate)
        return new_propagate


def make_process_kernel(propagate):
    @cuda.jit()
    def process_kernel(rng_states, neutrons, n_neutrons_per_thread, args):
        N = len(neutrons)
        thread_index = cuda.grid(1)
        start_index = thread_index*n_neutrons_per_thread
        end_index = min(start_index+n_neutrons_per_thread, N)
        for i in range(start_index, end_index):
            propagate(thread_index, rng_states, neutrons[i], *args)
        return
    return process_kernel

def make_process_ms_kernel(propagate, num_ms):
    # set a max register limit (some larger components will cause an error due to register use)
    @cuda.jit(max_registers=100)
    def process_kernel(rng_states, neutrons, n_neutrons_per_thread, args):
        N = len(neutrons)
        thread_index = cuda.grid(1)
        start_index = thread_index*n_neutrons_per_thread
        end_index = min(start_index+n_neutrons_per_thread, N)
        out_neutrons = cuda.local.array(shape=(num_ms, 10), dtype=NB_FLOAT)
        for i in range(start_index, end_index):
            propagate(thread_index, rng_states, out_neutrons, neutrons[i], *args)
        return
    return process_kernel

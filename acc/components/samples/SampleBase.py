"""
Requirement for a sample component
* component class
  - inherit from SampleBase
  - ctor must create self.propagate_params
* `propagate` method
  - first three arguments: `threadindex`, `rng_states`, `neutron`
  - other args: match comp.propagate_params
* at the end of the module, register the propagate method
"""

from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states
from mcvine.acc.config import get_numba_floattype, get_numpy_floattype
NB_FLOAT = get_numba_floattype()

from ... import config
from ...config import rng_seed
from ..StochasticComponentBase import StochasticComponentBase as base, make_process_kernel, make_process_ms_kernel
class SampleBase(base):

    def process_no_buffer(self, N, threads_per_block=None, ntotalthreads=None):
        import time
        t1 = time.time()
        threads_per_block = threads_per_block or config.threads_per_block
        ntotalthreads = ntotalthreads or config.ntotalthreads
        self.call_process_no_buffer(
            self.__class__.process_kernel_no_buffer, N,
            ntotalthreads, threads_per_block
        )
        t2 = time.time()
        print("call_process_no_buffer: ", t2-t1)
        return

    def call_process_no_buffer(
            self, process_kernel_no_buffer, N,
            ntotthreads=None, threads_per_block=None,
    ):
        # Use provided values or fall back to defaults in config
        threads_per_block = threads_per_block or config.threads_per_block
        ntotthreads = ntotthreads or config.ntotalthreads

        ntotthreads = min(N, int(ntotthreads))
        nblocks = math.ceil(ntotthreads / threads_per_block)
        actual_nthreads = threads_per_block * nblocks
        n_neutrons_per_thread = math.ceil(N / actual_nthreads)
        print("%s blocks, %s threads, %s neutrons per thread" % (
            nblocks, threads_per_block, n_neutrons_per_thread))
        rng_states = create_xoroshiro128p_states(actual_nthreads, seed=rng_seed)
        self.check_kernel_launch(process_kernel_no_buffer, threads_per_block,
                                 rng_states, N, n_neutrons_per_thread, self.propagate_params)
        process_kernel_no_buffer[nblocks, threads_per_block](
            rng_states, N, n_neutrons_per_thread, self.propagate_params)
        cuda.synchronize()
        return

    @classmethod
    def register_propagate_method(cls, propagate):
        new_propagate = cls._adjust_propagate_type(propagate)
        if cls.is_multiplescattering:
            cls.process_kernel = make_process_ms_kernel(new_propagate, cls.NUM_MULTIPLE_SCATTER)
        else:
            cls.process_kernel = make_process_kernel(new_propagate)
        cls.process_kernel_no_buffer = make_process_kernel_no_buffer(new_propagate)
        return new_propagate


def make_process_kernel_no_buffer(propagate):
    @cuda.jit()
    def process_kernel_no_buffer(rng_states, N, n_neutrons_per_thread, args):
        thread_index = cuda.grid(1)
        start_index = thread_index*n_neutrons_per_thread
        end_index = min(start_index+n_neutrons_per_thread, N)
        neutron = cuda.local.array(shape=10, dtype=NB_FLOAT)
        for i in range(start_index, end_index):
            propagate(thread_index, rng_states, neutron, *args)
        return
    return process_kernel_no_buffer

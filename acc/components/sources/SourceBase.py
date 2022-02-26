from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states

from ..ComponentBase import ComponentBase as base
class SourceBase(base):

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
        from ...config import rng_seed
        rng_states = create_xoroshiro128p_states(actual_nthreads, seed=rng_seed)
        process_kernel[nblocks, threads_per_block](
            rng_states, in_neutrons, n_neutrons_per_thread, *self.propagate_params)
        cuda.synchronize()
        return

template_for_process_kernel = """
@cuda.jit
def process_kernel(rng_states, neutrons, n_neutrons_per_thread, {param_str}):
    N = len(neutrons)
    thread_index = cuda.grid(1)
    start_index = thread_index*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    for i in range(start_index, end_index):
        propagate(thread_index, rng_states, neutrons[i], {param_str})
    return
"""

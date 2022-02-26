"""
Requirement for a component
* component class
  - inherit from ComponentBase
  - ctor must create self.propagate_params
* `propagate` method
  - first argument: `neutron`
  - other args: match comp.propagate_params
* process_kernel method: see template at the end of this module
"""

from numba import cuda
import math

from mcni.AbstractComponent import AbstractComponent
class ComponentBase(AbstractComponent):

    def process(self, neutrons):
        from time import time
        from mcvine.acc.config import get_numba_floattype, get_numpy_floattype
        from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
        t1 = time()
        neutron_array = neutrons_as_npyarr(neutrons)
        neutron_array.shape = -1, ndblsperneutron
        neutron_array_dtype_api = neutron_array.dtype
        neutron_array_dtype_int = get_numpy_floattype()
        needs_cast = neutron_array_dtype_api != neutron_array_dtype_int
        if needs_cast:
            neutron_array = neutron_array.astype(neutron_array_dtype_int)
        t2 = time()
        from ..config import ntotalthreads, threads_per_block
        self.call_process(
            self.__class__.process_kernel,
            neutron_array,
            ntotthreads=ntotalthreads, threads_per_block = threads_per_block,
        )
        t3 = time()
        if needs_cast:
            neutron_array = neutron_array.astype(neutron_array_dtype_api)
        good = neutron_array[:, -1] > 0
        neutrons.resize(int(good.sum()), neutrons[0])
        neutrons.from_npyarr(neutron_array[good])
        t4 = time()
        print(self.name, ":prepare input array: ", t2-t1)
        print(self.name, ":call_process: ", t3-t2)
        print(self.name, ":prepare output neutrons: ", t4-t3)
        return neutrons

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
        process_kernel[nblocks, threads_per_block](
            in_neutrons, n_neutrons_per_thread, *self.propagate_params)
        cuda.synchronize()
        return

template_for_process_kernel = """
@cuda.jit()
def process_kernel(neutrons, n_neutrons_per_thread, {param_str}):
    N = len(neutrons)
    thread_index = cuda.grid(1)
    start_index = thread_index*n_neutrons_per_thread
    end_index = min(start_index+n_neutrons_per_thread, N)
    for i in range(start_index, end_index):
        propagate(neutrons[i], {param_str})
    return
"""

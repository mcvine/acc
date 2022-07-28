# -*- python -*-
#

import math
from numba import cuda, void, int64, boolean
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type

from .SourceBase import SourceBase
from ...config import get_numba_floattype
from ... import vec3
NB_FLOAT = get_numba_floattype()

category = 'sources'

# need slightly different process_kernel... methods because it needs neutron index.
# probably should consider creating a base class for such components
def make_process_kernel_no_buffer(propagate):
    @cuda.jit()
    def process_kernel_no_buffer(rng_states, N, n_neutrons_per_thread, args):
        thread_index = cuda.grid(1)
        start_index = thread_index*n_neutrons_per_thread
        end_index = min(start_index+n_neutrons_per_thread, N)
        neutron = cuda.local.array(shape=10, dtype=NB_FLOAT)
        for i in range(start_index, end_index):
            propagate(thread_index, rng_states, i, neutron, *args)
        return
    return process_kernel_no_buffer

def make_process_kernel(propagate):
    @cuda.jit()
    def process_kernel(rng_states, neutrons, n_neutrons_per_thread, args):
        N = len(neutrons)
        thread_index = cuda.grid(1)
        start_index = thread_index*n_neutrons_per_thread
        end_index = min(start_index+n_neutrons_per_thread, N)
        for i in range(start_index, end_index):
            propagate(thread_index, rng_states, i, neutrons[i], *args)
        return
    return process_kernel


class NeutronFromStorage(SourceBase):

    requires_neutron_index_in_processing = True

    def __init__(self, name, path):
        """
        Initialize this component with a component name and the path to the neutron file.

        Parameters
        ----------

        path : str
            path to neutron file
        """
        self.name = name
        # read neutrons
        from mcni.neutron_storage import storage
        _storage = storage( path, 'r' )
        neutrons = _storage.read(asnpyarr=True, wrap=False)
        _storage.close()
        #
        self.propagate_params = (neutrons,)
        import mcni
        neutrons = mcni.neutron_buffer(1)
        self.process(neutrons)


    @cuda.jit(void(
        int64, xoroshiro128p_type[:],
        int64, NB_FLOAT[:],
        NB_FLOAT[:, :],
    ), device=True)
    def propagate(
            threadindex, rng_states,
            neutron_index, neutron,
            neutrons_from_storage
    ):
        N = len(neutrons_from_storage)
        for i in range(10):
            neutron[i] = neutrons_from_storage[neutron_index%N, i]
        return

    @classmethod
    def register_propagate_method(cls, propagate):
        new_propagate = cls._adjust_propagate_type(propagate)
        cls.process_kernel = make_process_kernel(new_propagate)
        cls.process_kernel_no_buffer = make_process_kernel_no_buffer(new_propagate)
        return new_propagate


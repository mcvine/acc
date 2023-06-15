import os
import numpy as np, numba
from numba import cuda, void, int64
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type
from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

from .SampleBase import SampleBase

category = 'samples'

def SampleAssemblyFromXml(samplexml):
    from . import loadScattererComposite
    composite = loadScattererComposite(samplexml)
    return factory(composite)

def sampleassembly_from_xml(name, samplexml):
    klass = SampleAssemblyFromXml(samplexml)
    return klass(name)

def factory(composite):
    from ...scatterers import scatter_func_factory
    methods = scatter_func_factory.render(composite)
    scatter = methods['scatter']
    class Composite(SampleBase):

        def __init__(self, name):
            self.name = name
            self.propagate_params = ()

            # Aim neutrons toward the sample to cause JIT compilation.
            import mcni
            neutrons = mcni.neutron_buffer(1)
            neutrons[0] = mcni.neutron(r=(0, 0, -1), v=(0, 0, 1), prob=1, time=0)
            self.process(neutrons)

        @cuda.jit(
            void(int64, xoroshiro128p_type[:], NB_FLOAT[:]),
            device=True, inline=True,
        )
        def propagate(threadindex, rng_states, neutron):
            return
    Composite.propagate = scatter
    Composite.register_propagate_method(scatter)
    return Composite

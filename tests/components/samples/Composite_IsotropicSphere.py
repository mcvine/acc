import os
samplexml = "sampleassemblies/isotropic_sphere/sampleassembly.xml"
from numba import cuda, void, int64
from numba.cuda.random import xoroshiro128p_type
from mcvine.acc.config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

from mcvine.acc.components.samples.SampleBase import SampleBase

# from mcvine.acc.components.samples.composite import SampleAssemblyFromXml
# Base = SampleAssemblyFromXml(samplexml)
# class Composite(Base): pass

from mcvine.acc.components.samples import loadScattererComposite
composite = loadScattererComposite(samplexml)

from mcvine.acc.scatterers import scatter_func_factory
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

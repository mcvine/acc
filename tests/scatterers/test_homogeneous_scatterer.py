import os
thisdir = os.path.dirname(__file__)
import numpy as np

def test():
    path = os.path.join(thisdir, "sampleassemblies", 'isotropic_sphere', 'sampleassembly.xml')
    from mcvine.acc.components.samples import loadFirstHomogeneousScatterer
    hs = loadFirstHomogeneousScatterer(path)
    shape = hs.shape()
    kernel = hs.kernel()
    mcweights = 1., 1., 1.
    packing_factor = 0.6
    from mcvine.acc.scatterers.homogeneous_scatterer import factory
    methods = factory(shape, kernel, mcweights, packing_factor)
    interact_path1 = methods['interact_path1']
    threadindex = 0
    rng_states = None
    N = 10000
    p = 0.
    absorbed = 0
    for i in range(N):
        neutron = np.array([0.0,0,0, 0,0,1000, 0,0, 0, 1.])
        interact_path1(threadindex, rng_states, neutron)
        p1 = neutron[-1]
        if p1>0:
            p+=p1
        else:
            absorbed += 1
    p/=N
    absorbed/=N
    assert p<.8 and p>.7
    assert absorbed < 0.36 and absorbed > 0.3
    return

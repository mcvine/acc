import os
thisdir = os.path.dirname(__file__)
import numpy as np
import pytest
from mcvine.acc import test

@pytest.mark.skipif(not test.USE_CUDASIM, reason='no CUDASIM')
def test():
    path = os.path.join(thisdir, "sampleassemblies", 'sample+2cylinders', 'sampleassembly.xml')
    from mcvine.acc.components.samples import loadScattererComposite
    composite = loadScattererComposite(path)
    from mcvine.acc.scatterers.composite_scatterer import factory_3
    methods = factory_3(composite)
    interact_path1 = methods['interact_path1']
    neutron = np.array([0.0,0,0, 0,0,1000, 0,0, 0, 1.])
    threadindex = 0
    rng_states = None
    N = 10 #000
    p = 0.
    absorbed = 0
    for i in range(N):
        neutron = np.array([0.0,0,0, 0,0,1000, 0,0, 0, 1.])
        interact_path1(threadindex, rng_states, neutron)
        print(neutron)
        p1 = neutron[-1]
        if p1>0:
            p+=p1
        else:
            absorbed += 1
    p/=N
    absorbed/=N
    print(p, absorbed)
    assert p < 0.1
    assert absorbed == 0
    return

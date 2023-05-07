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
    from mcvine.acc.scatterers import scatter_func_factory
    scatter_func_factory.render(composite)
    return

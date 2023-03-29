import os
from mcvine.acc.components.samples.homogeneous_single_scatterer import factory
thisdir = os.path.dirname(__file__)

path = os.path.join(thisdir, "sampleassemblies", 'Ni-sqekernel', 'sampleassembly.xml')
from mcvine.acc.components.samples import loadFirstHomogeneousScatterer
hs = loadFirstHomogeneousScatterer(path)
shape = hs.shape()
kernel = hs.kernel()
HSSbase = factory(shape = shape, kernel = kernel)
class HSS(HSSbase): pass

import os
from mcvine.acc.components.samples.homogeneous_single_scatterer import factory
thisdir = os.path.dirname(__file__)

path = os.path.join(thisdir, "sampleassemblies", 'isotropic_sphere', 'sampleassembly.xml')
from mcvine.acc.components.samples import loadFirstHomogeneousScatterer
hs = loadFirstHomogeneousScatterer(path)
shape = hs.shape()
HSSbase = factory(shape = shape, kernel = None)
class HSS(HSSbase): pass

import os
from mcvine.acc.components.samples.homogeneous_multiple_scatterer_test import factory
thisdir = os.path.dirname(__file__)

path = os.path.join(thisdir, "sampleassemblies", 'isotropic_sphere', 'sampleassembly.xml')
from mcvine.acc.components.samples import loadFirstHomogeneousScatterer
hs = loadFirstHomogeneousScatterer(path)
shape = hs.shape()
kernel = hs.kernel()
HMSbase = factory(shape = shape, kernel = kernel)
class HMS(HMSbase): pass

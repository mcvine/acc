import os
from mcvine.acc.components.samples.homogeneous_multiple_scatterer import factory
thisdir = os.path.dirname(__file__)

path = os.path.join(thisdir, "sampleassemblies", 'isotropic_hollowcylinder', 'sampleassembly.xml')
from mcvine.acc.components.samples import loadFirstHomogeneousScatterer
hs = loadFirstHomogeneousScatterer(path)
shape = hs.shape()
kernel = hs.kernel()
HMSbase = factory(shape = shape, kernel = kernel, max_ms_loops_path1=3)
class HMS(HMSbase): pass

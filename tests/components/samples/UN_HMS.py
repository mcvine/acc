import os
from mcvine.acc.components.samples.homogeneous_multiple_scatterer import factory
thisdir = os.path.dirname(__file__)

path = os.path.join(thisdir, "sampleassemblies", 'UN', 'sampleassembly.xml')
from mcvine.acc.components.samples import loadFirstHomogeneousScatterer
hs = loadFirstHomogeneousScatterer(path)
shape = hs.shape()
kernel = hs.kernel()
HMSbase = factory(shape = shape, kernel = kernel, max_ms_loops=1, max_ms_loops_path1=5, max_scattered_neutrons=33)
# HMSbase = factory(shape = shape, kernel = kernel, max_ms_loops=1, max_ms_loops_path1=1, max_scattered_neutrons=3)
class HMS(HMSbase): pass

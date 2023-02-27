import os, sys
thisdir = os.path.abspath(os.path.dirname(__file__))
if thisdir not in sys.path: sys.path.insert(0, thisdir)
import mcvine, mcvine.components as mc

def source(ctor, Ei):
    return ctor(
        name = 'src',
        radius = 0., width = 0.01, height = 0.01, dist = 1.,
        xw = 0.008, yh = 0.008,
        E0 = Ei, dE=0.01*Ei, Lambda0=0, dLambda=0.,
        flux=1, gauss=0
    )
source_cpu = lambda Ei: source(mc.sources.Source_simple, Ei)
from mcvine.acc.components.sources.source_simple import Source_simple
source_gpu = lambda Ei: source(Source_simple, Ei)

sample_xml=os.path.join(thisdir, "sampleassemblies", "isotropic_hollowcylinder", "sampleassembly.xml")
sample_cpu = lambda: mc.samples.SampleAssemblyFromXml('sample', sample_xml)
from HMS_isotropic_hollowcylinder import HMS
sample_gpu = lambda: HMS('sample')

def instrument(
        use_gpu,
        Ei=500.,
        source_cpu_factory=source_cpu,
        sample_cpu_factory=sample_cpu,
        source_gpu_factory=source_gpu,
        sample_gpu_factory=sample_gpu,
        z_sample=2.0
):
    instrument = mcvine.instrument()
    if use_gpu:
        instrument.append(source_gpu_factory(Ei), position=(0, 0, 0))
        sample = sample_gpu_factory()
        instrument.append(sample, position=(0, 0, z_sample))
    else:
        instrument.append(source_cpu_factory(Ei), position=(0, 0, 0))
        sample = sample_cpu_factory()
        instrument.append(sample, position=(0, 0, z_sample))
    return instrument

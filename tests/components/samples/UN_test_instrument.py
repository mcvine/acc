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

sample_xml=os.path.join(thisdir, "sampleassemblies", "UN", "sampleassembly.xml")
sample_cpu = lambda: mc.samples.SampleAssemblyFromXml('sample', sample_xml)
from UN_HSS import HSS
sample_gpu = lambda: HSS('sample')

def monitor(ctor, Ei, filename='iqe.dat'):
    return ctor(
        'iqe_monitor',
        Ei = Ei,
        Qmin=0., Qmax=30.0, nQ = 150,
        Emin=-100.0, Emax=400.0, nE = 100,
        min_angle_in_plane=10., max_angle_in_plane=135.,
        min_angle_out_of_plane=-26., max_angle_out_of_plane=26.,
        filename = filename
    )
monitor_cpu = lambda Ei: monitor(mc.monitors.IQE_monitor, Ei)
from mcvine.acc.components.monitors.iqe_monitor import IQE_monitor
monitor_gpu = lambda Ei: monitor(IQE_monitor, Ei, filename='iqe.h5')

def instrument(
        Ei=500.,
        source_factory=source_cpu,
        sample_factory=sample_cpu,
        monitor_factory=monitor_cpu,
):
    instrument = mcvine.instrument()
    instrument.append(source_factory(Ei), position=(0,0,0))
    sample = sample_factory()
    instrument.append(sample, position=(0,0,1.1))
    instrument.append(monitor_factory(Ei), position=(0,0,0), relativeTo=sample)
    return instrument

import os, sys
thisdir = os.path.abspath(os.path.dirname(__file__))
if thisdir not in sys.path: sys.path.insert(0, thisdir)
import mcvine, mcvine.components as mc
from UN_test_instrument import instrument, sample_cpu, sample_gpu, monitor_cpu, monitor_gpu

source_cpu = lambda path: mc.sources.NeutronFromStorage('src', path)
from mcvine.acc.components.sources.neutronfromstorage import NeutronFromStorage
source_gpu = lambda path: NeutronFromStorage('src', path)

L_mod2sample = 13.6

def monitor_gpu2(Ei):
    from mcvine.acc.components.monitors.dgs_iqe_monitor import IQE_monitor
    return IQE_monitor(
        'iqe', Ei=Ei, L0=L_mod2sample,
        Qmin=0., Qmax=30.0, nQ = 150,
        Emin=-100.0, Emax=400.0, nE = 100,
        min_angle_in_plane=10., max_angle_in_plane=135.,
        min_angle_out_of_plane=-26., max_angle_out_of_plane=26.,
        radius = 3., filename = 'iqe.h5'
     )

def instrument(
        Ei=512.,
        neutron_beam = os.path.join(thisdir, 'ARCS_vsource_512.mcv'),
        source_factory=source_cpu,
        sample_factory=sample_cpu,
        monitor_factory=monitor_cpu,
):
    instrument = mcvine.instrument()
    instrument.append(source_factory(neutron_beam), position=(0,0,0))
    sample = sample_factory()
    instrument.append(sample, position=(0,0,0.15))
    instrument.append(monitor_factory(Ei), position=(0,0,0), relativeTo=sample)
    return instrument

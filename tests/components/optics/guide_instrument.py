#!/usr/bin/env python

import mcvine, mcvine.components
mc = mcvine.components
from mcni import rng_seed
def seed(): return 0
rng_seed.seed = seed

def instrument(guide_mod=None, guide_factory=None, save_neutrons_after_guide=False):
    instrument = mcvine.instrument()
    source = mc.sources.Source_simple(
        name = 'source',
        radius = 0., width = 0.03, height = 0.03, dist = 1.,
        xw = 0.035, yh = 0.035,
        Lambda0 = 10., dLambda = 9.5,
    )
    instrument.append(source, position=(0,0,0.))
    if guide_mod:
        import importlib
        mod = importlib.import_module(guide_mod)
        # assume factory name is "Guide" in the given module
        guide_factory = mod.Guide
    elif guide_factory:
        guide_factory = eval(guide_factory)
    acc_guide = guide_factory(
        name = 'guide',
        w1=0.035, h1=0.035, w2=0.035, h2=0.035, l=10,
        R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003,
    )
    instrument.append(acc_guide, position=(0, 0, 1.))
    if save_neutrons_after_guide:
        after_guide = mc.monitors.NeutronToStorage(name='after_guide', path='after_guide.mcv')
        instrument.append(after_guide, position=(0, 0, 11.))
    Ixy = mc.monitors.PSD_monitor(
        name = 'Ixy', nx=250, ny=250, filename="Ixy.dat", xwidth=0.08, yheight=0.08,
        restore_neutron=True
    )
    instrument.append(Ixy, position=(0,0,12))
    Ixdivx = mc.monitors.DivPos_monitor(
        name = 'Ixdivx',
        npos=250, ndiv=250, filename="Ixdivx.dat", xwidth=0.08, yheight=0.08,
        restore_neutron=True
    )
    instrument.append(Ixdivx, position=(0,0,12))
    Iydivy = mc.monitors.DivPos_monitor(
        name = 'Iydivy',
        npos=250, ndiv=250, filename="Iydivy.dat", xwidth=0.08, yheight=0.08,
        restore_neutron=True
    )
    instrument.append(Iydivy, position=(0,0,12), orientation=(0, 0, 90))
    return instrument

#!/usr/bin/env python

import os
import mcvine, mcvine.components
mc = mcvine.components
from mcni import rng_seed
def seed(): return 0
rng_seed.seed = seed

thisdir = os.path.dirname(__file__)

def instrument(
        guide_mod=None, guide_factory=None,
        guide11_dat=None, guide11_len = 10.99, guide11_mx = 6., guide11_my = 6.,
        save_neutrons_before_guide=False,
        save_neutrons_after_guide=False):
    instrument = mcvine.instrument()
    source = mc.sources.Source_simple(
        name = 'source',
        radius = 0., width = 0.03, height = 0.03, dist = 6.35,
        xw = 0.35, yh = 0.35,
        Lambda0 = 10., dLambda = 9.5,
    )
    instrument.append(source, position=(0,0,0.))
    if save_neutrons_before_guide:
        before_guide = mc.monitors.NeutronToStorage(
            name='before_guide', path='before_tapered_guide.mcv')
        instrument.append(before_guide, position=(0, 0, 6.35))
    kwds = {}
    if guide_mod:
        import importlib
        mod = importlib.import_module(guide_mod)
        # assume factory name is "Guide" in the given module
        guide_factory = mod.Guide
        kwds = {'floattype': "float64"}
    elif guide_factory:
        guide_factory = eval(guide_factory)
    if guide11_dat is None:
        guide11_dat = os.path.join(thisdir, "data", "VERDI_V01_guide1.1")
    guide = guide_factory(
        name = 'guide',
        option="file={}".format(guide11_dat),
        l=guide11_len, mx=guide11_mx, my=guide11_my, **kwds
    )
    instrument.append(guide, position=(0, 0, 6.35))
    z_guide_end = 6.35+guide11_len
    if save_neutrons_after_guide:
        after_guide = mc.monitors.NeutronToStorage(
            name='after_guide', path='after_tapered_guide.mcv')
        instrument.append(after_guide, position=(0, 0, z_guide_end))
    Ixy = mc.monitors.PSD_monitor(
        name = 'Ixy', nx=250, ny=250, filename="Ixy.dat", xwidth=0.25, yheight=0.25,
        restore_neutron=True
    )
    instrument.append(Ixy, position=(0,0,z_guide_end+1e-3))
    Ixdivx = mc.monitors.DivPos_monitor(
        name = 'Ixdivx', filename="Ixdivx.dat",
        npos=250, xwidth=0.25, yheight=0.25,
        ndiv=250, maxdiv=0.5,
        restore_neutron=True
    )
    instrument.append(Ixdivx, position=(0,0,z_guide_end+1e-3))
    Iydivy = mc.monitors.DivPos_monitor(
        name = 'Iydivy', filename="Iydivy.dat",
        npos=250, xwidth=0.25, yheight=0.25,
        ndiv=250, maxdiv=0.5,
        restore_neutron=True
    )
    instrument.append(Iydivy, position=(0,0,z_guide_end+1e-3), orientation=(0, 0, 90))
    return instrument

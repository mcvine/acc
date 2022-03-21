#!/usr/bin/env python

import mcvine, mcvine.components
mc = mcvine.components
from mcni import rng_seed

rng_seed.seed = lambda: 0


def instrument(
        module=None, factory=None,
):
    instrument = mcvine.instrument()
    source = mc.sources.Source_simple(
        name='source',
        radius=0., width=0.03, height=0.03, dist=1.,
        xw=0.04, yh=0.04,
        Lambda0=10., dLambda=9.5
    )
    instrument.append(source, position=(0, 0, 0))
    if module:
        import importlib
        module = importlib.import_module(module)
        # assume factory name is "Beamstop" in the given module
        factory = module.Beamstop
    elif factory:
        factory = eval(factory)
    acc_beamstop = factory(
        name='beamstop',
        radius=0.015
    )
    instrument.append(acc_beamstop, position=(0, 0, 1))
    Ixy = mc.monitors.PSD_monitor(
        name='Ixy',
        nx=250, ny=250, filename="Ixy.dat", xwidth=0.10, yheight=0.10,
        restore_neutron=True
    )
    instrument.append(Ixy, position=(0, 0, 1.5))
    Ixdivx = mc.monitors.DivPos_monitor(
        name='Ixdivx',
        npos=250, ndiv=250, filename="Ixdivx.dat", xwidth=0.10, yheight=0.10,
        restore_neutron=True
    )
    instrument.append(Ixdivx, position=(0, 0, 1.5))
    Iydivy = mc.monitors.DivPos_monitor(
        name='Iydivy',
        npos=250, ndiv=250, filename="Iydivy.dat", xwidth=0.10, yheight=0.10,
        restore_neutron=True
    )
    instrument.append(Iydivy, position=(0, 0, 1.5), orientation=(0, 0, 90))
    return instrument

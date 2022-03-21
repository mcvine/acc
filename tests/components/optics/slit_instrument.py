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
        xw=0.035, yh=0.035,
        Lambda0=10., dLambda=9.5
    )
    instrument.append(source, position=(0, 0, 0))
    if module:
        import importlib
        mod = importlib.import_module(module)
        # assume factory name is "Slit" in the given module
        factory = mod.Slit
    elif factory:
        factory = eval(factory)
    acc_slit = factory(
        name='slit',
        width=0.02, height=0.01, cut=0.001
    )
    instrument.append(acc_slit, position=(0, 0, 1))
    Ixy = mc.monitors.PSD_monitor(
        name='Ixy',
        nx=250, ny=250, filename="Ixy.dat", xwidth=0.08, yheight=0.08,
        restore_neutron=True
    )
    instrument.append(Ixy, position=(0, 0, 1.5))
    Ixdivx = mc.monitors.DivPos_monitor(
        name='Ixdivx',
        npos=250, ndiv=250, filename="Ixdivx.dat", xwidth=0.08, yheight=0.08,
        restore_neutron=True
    )
    instrument.append(Ixdivx, position=(0, 0, 1.5))
    Iydivy = mc.monitors.DivPos_monitor(
        name='Iydivy',
        npos=250, ndiv=250, filename="Iydivy.dat", xwidth=0.08, yheight=0.08,
        restore_neutron=True
    )
    instrument.append(Iydivy, position=(0, 0, 1.5), orientation=(0, 0, 90))
    return instrument

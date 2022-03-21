#!/usr/bin/env python

import mcvine, mcvine.components
mc = mcvine.components

from mcni import rng_seed
rng_seed.seed = lambda: 0


def instrument(
        arm_mod=None, arm_factory=None,
):
    instrument = mcvine.instrument()
    source = mc.sources.Source_simple(
        name='source',
        radius=0., width=0.03, height=0.03, dist=12.,
        xw=0.035, yh=0.035,
        Lambda0=10., dLambda=9.5,
    )
    instrument.append(source, position=(0, 0, 0.))
    if arm_mod:
        import importlib
        mod = importlib.import_module(arm_mod)
        # assume factory name is "Arm" in the given module
        arm_factory = mod.Arm
    elif arm_factory:
        arm_factory = eval(arm_factory)
    acc_arm = arm_factory(
        name='arm'
    )
    instrument.append(acc_arm, position=(0, 0, 1.))
    Ixy = mc.monitors.PSD_monitor(
        name='Ixy', nx=250, ny=250, filename="Ixy.dat", xwidth=0.05,
        yheight=0.05,
        restore_neutron=True
    )
    instrument.append(Ixy, position=(0, 0, 12))
    return instrument

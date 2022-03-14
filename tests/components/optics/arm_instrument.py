#!/usr/bin/env python

import mcvine, mcvine.components
mc = mcvine.components

from mcni import rng_seed
rng_seed.seed = lambda: 0


def instrument(
        arm_mod=None, arm_factory=None,
        save_neutrons_before_arm=False, save_neutrons_after_arm=False,
):
    instrument = mcvine.instrument()
    source = mc.sources.Source_simple(
        name='source',
        radius=0., width=0.03, height=0.03, dist=1.,
        xw=0.035, yh=0.035,
        Lambda0=10., dLambda=9.5,
    )
    instrument.append(source, position=(0, 0, 0.))
    if save_neutrons_before_arm:
        before_arm = mc.monitors.NeutronToStorage(name='before_arm',
                                                  path='before_arm.mcv')
        instrument.append(before_arm, position=(0, 0, 1.))
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
    if save_neutrons_after_arm:
        after_arm = mc.monitors.NeutronToStorage(name='after_arm',
                                                 path='after_arm.mcv')
        instrument.append(after_arm, position=(0, 0, 11.))
    Ixy = mc.monitors.PSD_monitor(
        name='Ixy', nx=250, ny=250, filename="Ixy.dat", xwidth=0.08,
        yheight=0.08,
        restore_neutron=True
    )
    instrument.append(Ixy, position=(0, 0, 12))
    Ixdivx = mc.monitors.DivPos_monitor(
        name='Ixdivx',
        npos=250, ndiv=250, filename="Ixdivx.dat", xwidth=0.08, yheight=0.08,
        restore_neutron=True
    )
    instrument.append(Ixdivx, position=(0, 0, 12))
    Iydivy = mc.monitors.DivPos_monitor(
        name='Iydivy',
        npos=250, ndiv=250, filename="Iydivy.dat", xwidth=0.08, yheight=0.08,
        restore_neutron=True
    )
    instrument.append(Iydivy, position=(0, 0, 12), orientation=(0, 0, 90))
    return instrument

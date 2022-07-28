#!/usr/bin/env python

import mcvine, mcvine.components
mc = mcvine.components
from mcni import rng_seed
def seed(): return 0
rng_seed.seed = seed

def instrument(source_factory=None, all_acc_components=False):
    instrument = mcvine.instrument()
    # src
    source = source_factory(name = 'source')
    instrument.append(source, position=(0,0,0.))
    # mon
    kargs = dict(name = 'Ixy', nx=250, ny=250, filename="Ixy.dat", xwidth=0.05, yheight=0.05)
    if all_acc_components:
        from mcvine.acc.components.monitors.psd_monitor import PSD_monitor as factory
        kargs.update(dict(filename="Ixy.h5"))
    else:
        kargs.update(dict(restore_neutron=True))
        factory = mc.monitors.PSD_monitor
    Ixy = factory(**kargs)
    instrument.append(Ixy, position=(0,0,5))
    return instrument

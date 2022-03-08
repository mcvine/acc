#!/usr/bin/env python

import mcvine, mcvine.components
mc = mcvine.components
from mcni import rng_seed
def seed(): return 0
rng_seed.seed = seed

def instrument(
        source_factory=None,
):
    instrument = mcvine.instrument()
    source = source_factory(name = 'source')
    instrument.append(source, position=(0,0,0.))
    Ixy = mc.monitors.PSD_monitor(
        name = 'Ixy', nx=250, ny=250, filename="Ixy.dat", xwidth=0.05, yheight=0.05,
        restore_neutron=True
    )
    instrument.append(Ixy, position=(0,0,5))
    return instrument

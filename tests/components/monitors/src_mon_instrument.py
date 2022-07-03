#!/usr/bin/env python

import mcvine, mcvine.components
mc = mcvine.components
from mcni import rng_seed
def seed(): return 0
rng_seed.seed = seed

from mcvine.acc.components.sources.source_simple import Source_simple

def default_source():
    return Source_simple(
        "source",
        radius=0., height=0.03, width=0.03, dist=5.0,
        xw=0.03, yh=0.03,
        E0=0, dE=0, Lambda0=5., dLambda=2,
        flux=1, gauss=False, N=1
    )

def instrument(source_factory=None, monitor_factory=None, monitor_z=5):
    if monitor_factory is None:
        raise RuntimeError("missing monitor_factory!")
    instrument = mcvine.instrument()
    source_factory = source_factory or default_source
    source = source_factory()
    instrument.append(source, position=(0,0,0.))

    mon = monitor_factory()
    instrument.append(mon, position=(0,0,monitor_z))
    return instrument

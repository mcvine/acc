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
    source = mc.sources.Source_simple(
        'src',
        radius = 0., width = 0.03, height = 0.03, dist = 1.,
        xw = 0.035, yh = 0.035,
        Lambda0 = 10., dLambda = 9.5, E0=0., dE=0.0,
        flux=1, gauss=0
    )
    instrument.append(source, position=(0,0,0.))

    guide1 = mc.optics.Guide(
        'guide',
        w1=0.035, h1=0.035, w2=0.035, h2=0.035, l=10,
        R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003,
    )
    instrument.append(guide1, position=(0,0,1.))

    mon = mc.monitors.DivPos_monitor(
        'mon',
        xwidth=0.08, yheight=0.08,
        maxdiv=2.,
        npos=250, ndiv=251,
        filename="divpos.dat"
    )
    instrument.append(mon, position=(0,0,10+1+1.))
    return instrument

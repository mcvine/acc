#!/usr/bin/env python

import os, mcvine
from mcvine.acc.components.sources.source_simple import Source_simple
from HSS_isotropic_sphere import HSS
thisdir = os.path.dirname(__file__)

def instrument():
    instrument = mcvine.instrument()

    source = Source_simple(
        'src',
        radius = 0., width = 0.03, height = 0.03, dist = 1.,
        xw = 0.035, yh = 0.035,
        Lambda0 = 10., dLambda = 9.5, E0=0., dE=0.0,
        flux=1, gauss=0
    )
    instrument.append(source, position=(0,0,0.))

    sample = HSS('sample')
    instrument.append(sample, position=(0,0,1.))

    return instrument

#!/usr/bin/env python

import os, mcvine
from mcvine.acc.components.sources.source_simple import Source_simple
from HMS_test_isotropic_sphere import HMS
thisdir = os.path.dirname(__file__)

def instrument(monitor_factory=None, z_sample=1.):
    instrument = mcvine.instrument()

    source = Source_simple(
        'src',
        radius = 0., width = 0.01, height = 0.01, dist = 1.,
        xw = 0.008, yh = 0.008,
        Lambda0 = 10., dLambda = 9.5, E0=0., dE=0.0,
        flux=1, gauss=0
    )
    instrument.append(source, position=(0,0,0.))

    sample = HMS('sample')
    instrument.append(sample, position=(0,0,z_sample))

    if monitor_factory is not None:
        monitor = monitor_factory()
        instrument.append(monitor, position=(0,0,z_sample+0.01))

    return instrument

#!/usr/bin/env python

import os, mcvine
thisdir = os.path.dirname(__file__)

def instrument(sample_factory=None, monitor_factory=None, z_sample=2.0, source_factory=None):
    instrument = mcvine.instrument()
    if source_factory is None:
        from mcvine.acc.components.sources.source_simple import Source_simple
        source = Source_simple(
            'src',
            radius = 0., width = 0.01, height = 0.01, dist = 1.,
            xw = 0.008, yh = 0.008,
            Lambda0 = 10., dLambda = 9.5, E0=0., dE=0.0,
            flux=1, gauss=0
        )
    else:
        source = source_factory()
    instrument.append(source, position=(0,0,0.))

    if sample_factory is None:
        from HSS_isotropic_hollowcylinder import HSS
        sample = HSS('sample')
    else:
        sample = sample_factory()
    instrument.append(sample, position=(0,0,z_sample))

    if monitor_factory is not None:
        monitor = monitor_factory()
        instrument.append(monitor, position=(0,0,z_sample))

    return instrument

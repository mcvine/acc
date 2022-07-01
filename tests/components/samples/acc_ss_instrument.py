#!/usr/bin/env python

import os, mcvine
from mcvine.acc.components.sources.source_simple import Source_simple
from mcvine.acc.components.samples.homogeneous_single_scatterer import HomogeneousSingleScatterer
# from mcvine.acc.components.samples.homogeneous_single_scatterer import factory
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

    path = os.path.join(thisdir, "sampleassemblies", 'isotropic_sphere', 'sampleassembly.xml')
    from mcvine.acc.components.samples import loadFirstHomogeneousScatterer
    hs = loadFirstHomogeneousScatterer(path)
    shape = hs.shape()
    # sample = factory(shape = shape, kernel = None)('sample')
    sample = HomogeneousSingleScatterer('sample')
    instrument.append(sample, position=(0,0,1.))

    return instrument

#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

import mcvine
import mcvine.components as mc

def instrument(
        source2sample = 10.0,
):
    instrument = mcvine.instrument()

    source = mc.sources.Source_simple(
        'source',
        radius = 0, height = 0.0003, width = 0.0003,
        dist = source2sample-0.1,
        xw = 0.0003, yh = 0.0003,
        E0 = 0., dE = 0.,
        Lambda0 = 10, dLambda = 0.65,
        flux = 1.0, gauss = 0
    )
    instrument.append(source, position=(0,0,0))

    from HSS_sans2d_grid import HSS
    sample = HSS('sample')
    instrument.append(sample, position=(0,0,source2sample))

    Ixy = mc.monitors.PSD_monitor(
        'Ixy',
        nx = 250, ny = 250,
        filename = "Ixy.dat",
        xmin = -0.075, xmax = 0.075,
        ymin = -0.075, ymax = 0.075,
    )
    instrument.append(Ixy, position=(0,0,10), relativeTo=sample)
    return instrument

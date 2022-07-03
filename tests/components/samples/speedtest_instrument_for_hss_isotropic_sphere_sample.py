#!/usr/bin/env python

import os, sys, mcvine, mcvine.components as mcomps
thisdir = os.path.dirname(__file__)
if thisdir not in sys.path:
    sys.path.insert(0, thisdir)

def instrument(z_sample=1., use_acc_components=True):
    instrument = mcvine.instrument()

    if use_acc_components:
        from mcvine.acc.components.sources.source_simple import Source_simple as source_factory
    else:
        source_factory = mcomps.sources.Source_simple
    source = source_factory(
        'src',
        radius = 0., width = 0.01, height = 0.01, dist = 1.,
        xw = 0.008, yh = 0.008,
        Lambda0 = 10., dLambda = 9.5, E0=0., dE=0.0,
        flux=1, gauss=0
    )
    instrument.append(source, position=(0,0,0.))

    if use_acc_components:
        from HSS_isotropic_sphere import HSS
        sample = HSS('sample')
    else:
        path = os.path.join(thisdir, "sampleassemblies", 'isotropic_sphere', 'sampleassembly.xml')
        sample = mcomps.samples.SampleAssemblyFromXml('sample', path)
    instrument.append(sample, position=(0,0,z_sample))

    if use_acc_components:
        from mcvine.acc.components.monitors.psd_monitor_4pi import PSD_monitor_4Pi
        monitor = PSD_monitor_4Pi(
            "mon",
            nphi=30, ntheta=30, radius=3,
            filename = "psd_4pi.h5",
        )
    else:
        monitor = mcomps.monitors.PSD_monitor_4PI(
            "mon",
            nx=30, ny=30, radius=3,
            filename = "psd_4pi.h5",
        )
    instrument.append(monitor, position=(0,0,z_sample))

    return instrument

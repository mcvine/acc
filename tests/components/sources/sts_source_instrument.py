#!/usr/bin/env python

import os

thisdir = os.path.dirname(__file__)
dat = os.path.join(thisdir, 'BL20-CY-123D-STS-Min-2G-source_mctal-195_sp.dat')

def instrument(is_acc):
    import mcvine
    instrument = mcvine.instrument()
    if is_acc:
        from mcvine.acc.components.sources.STS_source import STS_source as factory
    else:
        import mcvine.components
        factory = mcvine.components.sources.STS_Source

    src = factory('src', filename=dat, Emin=3, Emax=82, xwidth=0.03, yheight=0.03, dist=2.5, focus_xw = .03, focus_yh = .035)
    instrument.append(src, position=(0,0,0))
    Ixy = mcvine.components.monitors.PSD_monitor(
        xwidth = 0.03, yheight = 0.035,
        nx = 10, ny = 10,
        filename = "Ixy.dat",
    )
    instrument.append(Ixy, position=(0, 0, 2.51))
    IE = mcvine.components.monitors.E_monitor(
        xwidth = 0.035, yheight = 0.04,
        Emin = 20, Emax = 70,
        nchan = 50,
        filename = "IE.dat",
    )
    instrument.append(IE, position=(0, 0, 2.52))
    return instrument

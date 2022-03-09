#!/usr/bin/env python

import os
import mcvine

from mcvine.acc.components.sources.source_simple import Source_simple
from mcvine.acc.components.optics.guide_tapering import Guide
from mcvine.acc.components.monitors.divpos_monitor import DivPos_monitor

from mcni import rng_seed
def seed(): return 0
rng_seed.seed = seed

thisdir = os.path.dirname(__file__)


def instrument(guide11_dat=None, guide11_len = 10.99, guide11_mx = 6., guide11_my = 6.):
    instrument = mcvine.instrument()
    source = Source_simple(
        name = 'source',
        radius = 0., width = 0.03, height = 0.03, dist = 6.35,
        xw = 0.35, yh = 0.35,
        Lambda0 = 10., dLambda = 9.5,
    )
    instrument.append(source, position=(0,0,0.))

    if guide11_dat is None:
        guide11_dat = os.path.join(thisdir, "data", "VERDI_V01_guide1.1")
    guide = Guide(
        name = 'guide',
        option="file={}".format(guide11_dat),
        l=guide11_len, mx=guide11_mx, my=guide11_my,
    )
    instrument.append(guide, position=(0, 0, 6.35))
    z_guide_end = 6.35+guide11_len

    Ixdivx = DivPos_monitor(
        name = 'Ixdivx', filename="Ixdivx.dat",
        npos=250, xwidth=0.25, yheight=0.25,
        ndiv=250, maxdiv=0.5,
    #    restore_neutron=True
    )
    instrument.append(Ixdivx, position=(0,0,z_guide_end+1e-3))
    '''
    # NOTE: having more than one component of the same type appears to not work with acc run_script
    Iydivy = DivPos_monitor(
        name = 'Iydivy', filename="Iydivy.dat",
        npos=250, xwidth=0.25, yheight=0.25,
        ndiv=250, maxdiv=0.5,
    #    restore_neutron=True
    )
    instrument.append(Iydivy, position=(0,0,z_guide_end+1e-3), orientation=(0, 0, 90))
    '''
    return instrument

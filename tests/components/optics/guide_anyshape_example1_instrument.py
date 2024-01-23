#!/usr/bin/env python

import os
from instrument_factory import construct

thisdir = os.path.dirname(__file__)

def instrument(is_acc=False):
    guide_length = 10.
    if is_acc:
        from mcvine.acc.components.optics.guide_anyshape import Guide_anyshape
        target = Guide_anyshape(
            name='guide',
            xwidth=0, yheight=0, zdepth=0,
            center=False, 
            R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003,
            geometry=os.path.join(thisdir, './data/guide_anyshape_straight_3.5cmX3.5cmX10m.off'),
        )
    else:
        import mcvine.components
        factory = mcvine.components.optics.Guide
        target = factory(
            name='guide',
            w1=0.035, h1=0.035, w2=0.035, h2=0.035, l=guide_length,
            R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003
        )
    return construct(target, guide_length)

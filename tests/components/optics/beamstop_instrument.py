#!/usr/bin/env python

from instrument_factory import construct


def instrument(is_acc):
    if is_acc:
        from mcvine.acc.components.optics.beamstop import Beamstop
        factory = Beamstop
    else:
        import mcvine.components
        factory = mcvine.components.optics.Beamstop

    target = factory(name='beamstop', radius=0.015)
    return construct(target, 0.)

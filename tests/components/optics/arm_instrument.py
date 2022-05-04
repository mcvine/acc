#!/usr/bin/env python

from instrument_factory import construct


def instrument(is_acc):
    if is_acc:
        from mcvine.acc.components.optics.arm import Arm
        factory = Arm
    else:
        import mcvine.components
        factory = mcvine.components.optics.Arm

    target = factory(name='arm')
    return construct(target, 0.)

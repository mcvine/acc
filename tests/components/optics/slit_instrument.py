#!/usr/bin/env python

from instrument_factory import construct


def instrument(is_acc):
    if is_acc:
        from mcvine.acc.components.optics.slit import Slit
        factory = Slit
    else:
        import mcvine.components
        factory = mcvine.components.optics.Slit

    target = factory(
        name='slit',
        width=0.02, height=0.01, cut=0.001
    )
    return construct(target, 0.)

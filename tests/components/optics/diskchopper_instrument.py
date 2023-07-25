#!/usr/bin/env python

from instrument_factory import construct


def instrument(is_acc, **kwds):
    if is_acc:
        from mcvine.acc.components.optics.disk_chopper import DiskChopper
        factory = DiskChopper
    else:
        import mcvine.components
        factory = mcvine.components.optics.DiskChopper_v2

    target = factory(name='diskchopper', theta_0=8, radius=0.35,
                     yheight=0.045, nu=-300,
                     nslit=1, 
                     jitter=1E-06, 
                     **kwds)

    #target = factory(**kwds)
    print(target)
    return construct(target, 0., gap=0.025 * 0.5)

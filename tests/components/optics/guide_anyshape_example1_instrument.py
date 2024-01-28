#!/usr/bin/env python

import os
from instrument_factory import construct

thisdir = os.path.dirname(__file__)

def instrument(
        is_acc=False, geometry=None, dims=None,
        acc_component_factory=None,
        nonacc_component_factory=None,
    ):
    guide_length = 10.
    if is_acc:
        geometry = geometry or os.path.join(thisdir, './data/guide_anyshape_straight_3.5cmX3.5cmX10m.off')
        if acc_component_factory is None:
            from mcvine.acc.components.optics.guide_anyshape import Guide_anyshape as factory
        else:
            factory = _import_factory(acc_component_factory)
        print(factory)
        target = factory(
            name='guide',
            xwidth=0, yheight=0, zdepth=0,
            center=False,
            R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003,
            geometry=geometry,
        )
    else:
        import mcvine.components
        if nonacc_component_factory is None:
            factory = nonacc_component_factory or mcvine.components.optics.Guide
        else:
            factory = eval(nonacc_component_factory)
        print(factory)
        if dims is None:
            dims = dict(
                w1=0.035, h1=0.035, w2=0.035, h2=0.035, l=guide_length,
            )
        target = factory(
            name='guide',
            R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003,
            G = 9.80665,
            **dims,
        )
    return construct(target, guide_length)

def _import_factory(factory):
    import importlib
    module = '.'.join(factory.split('.')[:-1])
    mod = importlib.import_module(module)
    return getattr(mod, factory.split('.')[-1])
 
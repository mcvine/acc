#!/usr/bin/env python

from test_sample_instrument_factory import construct


def instrument():
    # import mcvine.components
    from mcvine.acc.components.samples.isotropic_box import IsotropicBox as factory
    target = factory(
        name='sample',
        xwidth=0.01, yheight=0.01, zthickness=0.01,
        mu=10, sigma=10,
    )
    return construct(target, size=0.)

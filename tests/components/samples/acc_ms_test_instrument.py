#!/usr/bin/env python

import os, sys, mcvine
from mcvine.acc.components.sources.source_simple import Source_simple
thisdir = os.path.dirname(__file__)
if thisdir not in sys.path:
    sys.path.insert(0, thisdir)

def instrument(sample_factory=None, monitor_factory=None, z_sample=2.0):
    from acc_ss_test_instrument import instrument
    if sample_factory is None:
        def sample_factory():
            from HMS_isotropic_hollowcylinder import HMS
            return HMS('sample')
    return instrument(sample_factory, monitor_factory, z_sample)

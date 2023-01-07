#!/usr/bin/env python

import os, sys, mcvine
thisdir = os.path.dirname(__file__)
if thisdir not in sys.path:
    sys.path.insert(0, thisdir)

def instrument(sample_factory=None, monitor_factory=None, z_sample=2.0, source_factory=None):
    from acc_ss_test_instrument import instrument
    if sample_factory is None:
        def sample_factory():
            from HMS_isotropic_hollowcylinder import HMS
            return HMS('sample')
    return instrument(
        source_factory=source_factory,
        sample_factory=sample_factory,
        monitor_factory=monitor_factory,
        z_sample=z_sample,
    )

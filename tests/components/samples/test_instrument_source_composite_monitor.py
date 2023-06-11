#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

from test_sample_instrument_factory import construct

def instrument(samplexml, is_acc=True):
    if is_acc:
        from mcvine.acc.components.samples import loadScattererComposite
        composite = loadScattererComposite(samplexml)
        from mcvine.acc.components.samples.composite import factory
        target = factory(composite)("sample")
    else:
        import mcvine.components as mc
        target = mc.samples.SampleAssemblyFromXml("sample", samplexml)
    return construct(target, size=0.)

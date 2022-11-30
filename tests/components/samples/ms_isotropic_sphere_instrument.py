#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

from test_sample_instrument_factory import construct

def instrument(is_acc=True):
    if is_acc:
        from HMS_isotropic_sphere import HMS
        target = HMS(name='sample')
    else:
        import mcvine.components as mc
        xml=os.path.join(thisdir, "sampleassemblies", "isotropic_sphere", "sampleassembly.xml")
        target = mc.samples.SampleAssemblyFromXml("sample", xml)
    return construct(target, size=0., z_sample=2.0)

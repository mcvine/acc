#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

from test_sample_instrument_factory import construct

def instrument(is_acc=True):
    if is_acc:
        from mcvine.acc.components.samples.isotropic_box import IsotropicBox as factory
        from HSS_isotropic_sphere import HSS
        target = HSS(name='sample')
    else:
        import mcvine.components as mc
        xml=os.path.join(thisdir, "sampleassemblies", "isotropic_sphere", "sampleassembly.xml")
        target = mc.samples.SampleAssemblyFromXml("sample", xml)
    return construct(target, size=0.)

#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

from test_sample_instrument_factory import construct

def instrument(is_acc=True):
    if is_acc:
        from mcvine.acc.components.samples.isotropic_box import IsotropicBox as factory
        target = factory(
            name='sample',
            xwidth=0.01, yheight=0.01, zthickness=0.01,
            mu=10, sigma=10,
        )
    else:
        import mcvine.components as mc
        xml=os.path.join(thisdir, "sampleassemblies", "isotropic_box", "sampleassembly.xml")
        target = mc.samples.SampleAssemblyFromXml("sample", xml)
    return construct(target, size=0.)

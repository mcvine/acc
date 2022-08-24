#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

from test_powderdiffraction_sample_instrument_factory import construct, Builder

def instrument(is_acc=True):
    if is_acc:
        from HSS_fccAl_constantqe_box import HSS
        target = HSS(name='sample')
    else:
        import mcvine.components as mc
        xml=os.path.join(thisdir, "sampleassemblies", "Al-constantqekernel", "sampleassembly.xml")
        target = mc.samples.SampleAssemblyFromXml("sample", xml)
    source_params = dict(Lambda0=2, dLambda=0.01)
    return construct(target, size=0., source_params=source_params, builder=Builder())

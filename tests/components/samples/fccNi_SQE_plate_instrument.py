#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

from test_powderdiffraction_sample_instrument_factory import construct, Builder

def instrument(is_acc=True):
    if is_acc:
        from HSS_fccNi_SQE_plate import HSS
        target = HSS(name='sample')
    else:
        import mcvine.components as mc
        xml=os.path.join(thisdir, "sampleassemblies", "Ni-sqekernel", "sampleassembly.xml")
        target = mc.samples.SampleAssemblyFromXml("sample", xml)
    source_params = dict(E0 = 70.0, dE=0.1, Lambda0=0, dLambda=0.)
    return construct(
        target, size=0.,
        source_params=source_params, monitors=['PSD_4PI'],
        builder=Builder())

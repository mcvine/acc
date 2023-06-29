#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

from test_sample_instrument_factory import construct
from test_DGSSXRes_sample_instrument_factory import Builder
from mcni import neutron

def instrument(is_acc=True):
    if is_acc:
        from HSS_fccAl_DGSSXRes_plate import HSS
        target = HSS(name='sample')
    else:
        import mcvine.components as mc
        xml=os.path.join(thisdir, "sampleassemblies", "Al-DGSSXResKernel", "sampleassembly.xml")
        target = mc.samples.SampleAssemblyFromXml("sample", xml)
    source_params = dict(neutron=neutron([0., 0., 0.], [0., 0., 3000.0], prob=1.0, time=0.0))
    return construct(
        target, size=0., z_sample=6.0,
        source_params=source_params, monitors=[],
        builder=Builder(),
        save_neutrons_after=True)

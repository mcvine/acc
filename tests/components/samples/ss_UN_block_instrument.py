#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

from test_powdersqesample_instrument_factory import construct, Builder as BuilderBase

Ei = 400.0
class Builder(BuilderBase):
    def addIQEMonitor(self):
        return super(Builder, self).addIQEMonitor(
            Ei=Ei,
            Qmin=0., Qmax=20.0, nQ = 200,
            Emin=-200.0, Emax=600.0, nE = 120,
        )

def instrument(is_acc=True):
    if is_acc:
        from HSS_UN_box import HSS
        target = HSS(name='sample')
    else:
        import mcvine.components as mc
        xml=os.path.join(thisdir, "sampleassemblies", "UN", "sampleassembly.xml")
        target = mc.samples.SampleAssemblyFromXml("sample", xml)
    source_params = dict(E0 = Ei, dE=0.05*Ei, Lambda0=0, dLambda=0.)
    return construct(
        target, size=0.,
        source_params=source_params, monitors=['IQE'],
        builder=Builder())

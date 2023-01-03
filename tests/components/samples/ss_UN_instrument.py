#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

from test_powdersqesample_instrument_factory import construct, Builder as BuilderBase

Ei = 500.0
class Builder(BuilderBase):
    def addIQEMonitor(self):
        return super(Builder, self).addIQEMonitor(
            Ei=Ei,
            Qmin=0., Qmax=30.0, nQ = 150,
            Emin=-100.0, Emax=400.0, nE = 100,
        )

def instrument(is_acc=True):
    if is_acc:
        from UN_HSS import HSS
        target = HSS(name='sample')
    else:
        import mcvine.components as mc
        xml=os.path.join(thisdir, "sampleassemblies", "UN", "sampleassembly.xml")
        target = mc.samples.SampleAssemblyFromXml("sample", xml)
    source_params = dict(E0 = Ei, dE=0.01*Ei, Lambda0=0, dLambda=0.)
    return construct(
        target, size=0.,
        source_params=source_params, monitors=['IQE'],
        builder=Builder())

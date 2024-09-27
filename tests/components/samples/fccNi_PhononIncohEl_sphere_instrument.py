#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

from test_sample_instrument_factory import construct, Builder as base

class Builder(base):

    def addIQEMonitor(self, **kwds):
        params = dict(
            Ei = 70.0,
            Qmin=0., Qmax=6.0, nQ = 160,
            Emin=-.15, Emax=0.15, nE = 120,
            min_angle_in_plane=0., max_angle_in_plane=359.,
            min_angle_out_of_plane=-90., max_angle_out_of_plane=90.,
        )
        params.update(kwds)
        mon = self.get_monitor(
            subtype="IQE_monitor", name = "IQE",
            **params
            )
        self.add(mon, gap=0)


def instrument(is_acc=True):
    if is_acc:
        from HSS_fccNi_PhononIncohEl_sphere import HSS
        target = HSS(name='sample')
    else:
        import mcvine.components as mc
        xml=os.path.join(thisdir, "sampleassemblies", "incoh-el", "sampleassembly.xml")
        target = mc.samples.SampleAssemblyFromXml("sample", xml)
    source_params = dict(E0 = 70.0, dE=0.1, Lambda0=0, dLambda=0.)
    return construct(
        target, size=0.,
        source_params=source_params, monitors=['IQE'],
        builder=Builder())

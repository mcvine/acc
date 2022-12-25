# -*- python -*-
from test_sample_instrument_factory import construct, Builder as base

class Builder(base):

    def addIQEMonitor(self, **kwds):
        params = dict(
            Ei = 70.0,
            Qmin=0., Qmax=8.0, nQ = 160,
            Emin=-60.0, Emax=60.0, nE = 120,
            min_angle_in_plane=0., max_angle_in_plane=359.,
            min_angle_out_of_plane=-90., max_angle_out_of_plane=90.,
        )
        params.update(kwds)
        print(params)
        mon = self.get_monitor(
            subtype="IQE_monitor", name = "IQE",
            **params
            )
        self.add(mon, gap=0)

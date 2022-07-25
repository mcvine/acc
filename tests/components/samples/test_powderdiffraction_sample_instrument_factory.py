# -*- python -*-
from test_sample_instrument_factory import construct, Builder as base

class Builder(base):

    def addPSD_4PIMonitor(self):
        mon = self.get_monitor(
            subtype="PSD_monitor_4PI", name = "psd_4pi",
            nx=100, ny=100, radius=3)
        self.add(mon, gap=0)

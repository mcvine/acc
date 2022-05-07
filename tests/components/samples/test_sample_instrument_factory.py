# -*- python -*-

def construct(
        component, size, gap=0.,
        monitors=["PSD_4PI"],
        **kwds
):
    builder = Builder()
    from mcvine.acc.test.instrument_factory import construct
    return construct(component, size, gap, monitors=monitors, builder=builder, **kwds)

from mcvine.acc.test.instrument_factory import InstrumentBuilder as base

class Builder(base):
    def addPSD_4PIMonitor(self):
        mon = self.get_monitor(
            subtype="PSD_monitor_4PI", name = "psd_4pi",
            nx=180, ny=360, radius=3)
        self.add(mon)



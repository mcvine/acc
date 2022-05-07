# -*- python -*-

def construct(
        component, size, gap=1.,
        monitors=["Ixy", "Ixdivx", "Ixdivy"],
        **kwds
):
    builder = Builder()
    from mcvine.acc.test.instrument_factory import construct
    return construct(component, size, gap, monitors=monitors, builder=builder, **kwds)

from mcvine.acc.test.instrument_factory import InstrumentBuilder as base

class Builder(base):
    def addIxyMonitor(self):
        mon = self.get_monitor("PSD_monitor", "Ixy", nx=250, ny=250)
        self.add(mon)
    def addIxdivxMonitor(self):
        mon = self.get_monitor("DivPos_monitor", "Ixdivx", npos=250, ndiv=250)
        self.add(mon)
    def addIxdivyMonitor(self):
        mon = self.get_monitor("DivPos_monitor", "Ixdivy", npos=250, ndiv=250)
        self.add(mon, is_rotated=True)



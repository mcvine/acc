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

    @classmethod
    def get_source(cls):
        """
        Construct a source.
        """
        import mcvine.components as mc
        return mc.sources.Source_simple(
            name='source',
            radius=0., width=0.01, height=0.01, dist=1.,
            xw=0.008, yh=0.008,
            Lambda0=10., dLambda=9.5
        )

    def addPSD_4PIMonitor(self):
        mon = self.get_monitor(
            subtype="PSD_monitor_4PI", name = "psd_4pi",
            nx=90, ny=90, radius=3)
        self.add(mon)



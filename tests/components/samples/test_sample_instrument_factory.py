# -*- python -*-

def construct(
        component, size, z_sample=1.,
        monitors=["PSD_4PI"],
        save_neutrons_before=False, save_neutrons_after=False,
        source_params=None,
        builder = None,
):
    """
    implementation revised from mcvine.acc.tests.instrument_factory.construct
    """
    builder = builder or Builder()
    # source
    source_params = source_params or dict()
    builder.add(builder.get_source(**source_params), gap=z_sample)

    # instrument
    if save_neutrons_before:
        import mcvine.components as mc
        before = mc.monitors.NeutronToStorage(
            name=f"before_{component.name}",
            path=f"before_{component.name}.mcv")
        builder.add(before, gap=0)
    builder.add(component, gap=0)
    if save_neutrons_after:
        import mcvine.components as mc
        after = mc.monitors.NeutronToStorage(
            name=f"after_{component.name}",
            path=f"after_{component.name}.mcv")
        builder.add(after, gap=0)

    # monitors
    for mon in monitors:
        method = getattr(builder, f'add{mon}Monitor')
        method()
    return builder.instrument

from mcvine.acc.test.instrument_factory import InstrumentBuilder as base

class Builder(base):

    @classmethod
    def get_source(
            cls, Lambda0=10., dLambda=9.5, E0=0, dE=0,
    ):
        """
        Construct a source.
        """
        import mcvine.components as mc
        return mc.sources.Source_simple(
            name='source',
            radius=0., width=0.01, height=0.01, dist=0.9,
            xw=0.008, yh=0.008,
            Lambda0 = Lambda0, dLambda = dLambda,
            E0=E0, dE=dE,
        )

    def addPSD_4PIMonitor(self):
        mon = self.get_monitor(
            subtype="PSD_monitor_4PI", name = "psd_4pi",
            nx=30, ny=30, radius=3)
        self.add(mon, gap=0)

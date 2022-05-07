#!/usr/bin/env python

# Copyright (c) 2021-2022 by UT-Battelle, LLC.

import mcvine.components
mc = mcvine.components

from mcni import rng_seed
rng_seed.seed = lambda: 0

class InstrumentBuilder:
    """
    Helper for constructing an instrument as a sequence of components positioned
    along increasing z. Methods can have more arguments and complexity added
    as need for flexibility arises.
    """

    def __init__(self):
        self.instrument = mcvine.instrument()
        self.z = 0.

    @classmethod
    def get_source(cls):
        """
        Construct a source.
        """
        return mc.sources.Source_simple(
            name='source',
            radius=0., width=0.03, height=0.03, dist=1.,
            xw=0.035, yh=0.035,
            Lambda0=10., dLambda=9.5
        )

    @classmethod
    def get_monitor(cls, subtype, name, **kwargs):
        """
        Construct a monitor.
        Extra keyword arguments are passed to the given subtype.

        Parameters:
        subtype (str): which kind of monitor, e.g., "PSD_monitor"
        name (str): the name of the monitor and its files, e.g., "Ixy"
        """
        return getattr(mc.monitors, subtype)(
            name=name, filename=name+".dat",
            xwidth=0.1, yheight=0.1,
            restore_neutron=True,
            **kwargs
        )

    def add(self, component, gap=0., is_rotated=False):
        """
        Add a component to the instrument.

        Parameters:
        component: the component to add
        gap (float): how much space along z to leave between this and the next
        is_rotated (bool): if to rotate this instrument in positioning it
        """
        self.instrument.append(component, position=(0., 0., self.z),
                               orientation=(0., 0., 90. if is_rotated else 0.))
        self.z += gap


def construct(component, size, gap=1.,
              monitors=["Ixy", "Ixdivx", "Ixdivy"],
              save_neutrons_before=False, save_neutrons_after=False):
    """
    Construct an instrument.

    Parameters:
    component: the optics of interest, to place between sources and monitors
    size (float): the size of the component along the z-axis
    gap (float): how much space between the source, optics, and monitors
    monitors (list): the names of the monitors to include, e.g., "Ixy"
    save_neutrons_before (bool): if to save neutrons before the given optics
    save_neutrons_after (bool): if to save neutrons after the given optics
    """
    builder = InstrumentBuilder()

    # source
    builder.add(builder.get_source(), gap=gap)

    # instrument
    if save_neutrons_before:
        before = mc.monitors.NeutronToStorage(
            name=f"before_{component.name}",
            path=f"before_{component.name}.mcv")
        builder.add(before)
    if save_neutrons_after:
        builder.add(component, gap=size)
        after = mc.monitors.NeutronToStorage(
            name=f"after_{component.name}",
            path=f"after_{component.name}.mcv")
        builder.add(after, gap=gap)
    else:
        builder.add(component, gap=size+gap)

    # monitors
    if "Ixy" in monitors:
        builder.add(builder.get_monitor("PSD_monitor", "Ixy",
                                  nx=250, ny=250))
    if "Ixdivx" in monitors:
        builder.add(builder.get_monitor("DivPos_monitor", "Ixdivx",
                                  npos=250, ndiv=250))
    if "Ixdivy" in monitors:
        builder.add(builder.get_monitor("DivPos_monitor", "Ixdivy",
                                  npos=250, ndiv=250), is_rotated=True)

    return builder.instrument

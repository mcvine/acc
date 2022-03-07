#!/usr/bin/env python
#
# Copyright (c) 2021-2022 by UT-Battelle, LLC.

from numba import cuda, void

from ..ComponentBase import ComponentBase
from ...neutron import absorb

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

category = 'optics'


@cuda.jit(
    void(
        NB_FLOAT[:],
        NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT,
    ), device=True
)
def propagate(
        neutron,
        xmin, xmax, ymin, ymax, radius_squared, cut
):
    x, y, z, vx, vy, vz = neutron[:6]

    # check that neutron reaches z==0
    if z < 0 and vz <= 0 or z > 0 and vz >= 0:
        # absorb(neutron)
        return

    t, prob = neutron[-2:]

    # check that neutron weight makes the cut
    if prob < cut:
        absorb(neutron)
        return

    # bring neutron to z==0
    if z != 0:
        dt = -z / vz
        x += vx * dt
        y += vy * dt
        t += dt

    # check that neutron is not blocked by beamstop
    if radius_squared != 0:
        if x * x + y * y <= radius_squared:
            absorb(neutron)
            return
    else:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            absorb(neutron)
            return

    # neutron is not blocked by beamstop
    neutron[:3] = x, y, 0.
    neutron[-2] = t


@cuda.jit
def propagate_kernel(beamstop_nature, neutron):
    (xmin, xmax, ymin, ymax, radius_squared, cut) = beamstop_nature
    propagate(neutron, xmin, xmax, ymin, ymax, radius_squared, cut)


class Beamstop(ComponentBase):

    def __init__(
            self, name,
            xmin=0, xmax=0, ymin=0, ymax=0, radius=0, cut=0, width=0, height=0):
        """
        Initialize this beamstop component.
        The beamstop is at z==0.
        Setting radius to non-zero switches this beamstop from being rectangular
        to a circle centered on x,y==0,0.

        Parameters:
        name (str): the name of this component
        xmin (m): the smallest value of x that still passes a rectangular
            beamstop
        xmax (m): the largest value of x that still passes a rectangular
            beamstop
        ymin (m): the smallest value of y that still passes a rectangular
            beamstop
        ymax (m): the largest value of y that still passes a rectangular
            beamstop
        radius (m): the largest distance on x-y from 0,0 that still passes a
            circular beamstop
        cut: the smallest neutron weight that can pass the beamstop
        width (m): sets xmin and xmax to a width centered on x==0
        height (m): sets ymin and ymax to a width centered on y==0
        """
        self.name = name

        # Check and process the arguments.
        if width > 0:
            xmax = width / 2
            xmin = -xmax
        if height > 0:
            ymax = height / 2
            ymin = -ymax
        if not (xmin or xmax or ymin or ymax or radius):
            raise ValueError("a beamstop must have some extent")

        # Note the configuration of the beamstop.
        self.propagate_params = (
            float(xmin), float(xmax), float(ymin), float(ymax),
            float(radius * radius), float(cut)
        )

        # Aim neutrons toward the beamstop to cause JIT compilation.
        import mcni
        neutrons = mcni.neutron_buffer(2)
        # first neutron is absorbed
        neutrons[0] = mcni.neutron(r=(0, 0, -1), v=(0, 0, 1), prob=1, time=0)
        # second neutron passes by
        x_edge = radius if radius > 0 else xmax
        neutrons[1] = mcni.neutron(
            r=(x_edge * 2, 0, -1), v=(0, 0, 1), prob=1, time=0)
        self.process(neutrons)


Beamstop.register_propagate_method(propagate)

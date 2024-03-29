#!/usr/bin/env python
#
# Copyright (c) 2021-2022 by UT-Battelle, LLC.

import numpy as np
from numba import cuda, void

from ..ComponentBase import ComponentBase
from ...neutron import absorb

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

category = 'optics'


class Slit(ComponentBase):

    def __init__(
            self, name,
            xmin=0, xmax=0, ymin=0, ymax=0, radius=0, cut=0, width=0, height=0,
            **kwargs):
        """
        Initialize this Slit component.
        The slit is at z==0.
        Setting radius to non-zero switches this slit from being rectangular to
        a circle centered on x,y==0,0.

        Parameters:
        name (str): the name of this component
        xmin (m): the smallest value of x that still passes a rectangular slit
        xmax (m): the largest value of x that still passes a rectangular slit
        ymin (m): the smallest value of y that still passes a rectangular slit
        ymax (m): the largest value of y that still passes a rectangular slit
        radius (m): the largest distance on x-y from 0,0 that still passes a
            circular slit
        cut: the smallest neutron weight that can penetrate the slit
        width (m): sets xmin and xmax to a width centered on x==0
        height (m): sets ymin and ymax to a width centered on y==0
        """
        super().__init__(__class__, **kwargs)

        self.name = name

        # Check and process the arguments.
        if width > 0:
            xmax = width / 2
            xmin = -xmax
        if height > 0:
            ymax = height / 2
            ymin = -ymax
        if not (xmin or xmax or ymin or ymax or radius):
            raise ValueError("a slit must have some extent")

        # Note the configuration of the slit.
        self.propagate_params = (
            np.array([xmin, xmax, ymin, ymax, radius * radius, cut]),
        )

        # Aim neutrons toward the slit to cause JIT compilation.
        import mcni
        neutrons = mcni.neutron_buffer(2)
        # first neutron passes through
        neutrons[0] = mcni.neutron(r=(0, 0, -1), v=(0, 0, 1), prob=1, time=0)
        # second neutron is absorbed
        x_edge = radius if radius > 0 else xmax
        neutrons[1] = mcni.neutron(
            r=(x_edge * 2, 0, -1), v=(0, 0, 1), prob=1, time=0)
        self.process(neutrons)

    @cuda.jit(void(NB_FLOAT[:], NB_FLOAT[:]), device=True)
    def propagate(neutron, param_arr):
        xmin, xmax, ymin, ymax, radius_squared, cut = param_arr
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

        # check that neutron penetrates slit
        if radius_squared != 0:
            if x * x + y * y > radius_squared:
                absorb(neutron)
                return
        else:
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                absorb(neutron)
                return

        # neutron penetrates slit
        neutron[:2] = x, y
        neutron[2] = 0.
        neutron[-2] = t


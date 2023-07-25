#!/usr/bin/env python
#
# Copyright (c) 2021-2023 by UT-Battelle, LLC.

import numpy as np
from numba import cuda, void, int64
import math

from ..StochasticComponentBase import StochasticComponentBase
from ...neutron import absorb, prop_z0
from ...random import rand01, randpm1, randnorm

from ...config import get_numba_floattype
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type

NB_FLOAT = get_numba_floattype()
RAD2DEG = 180./math.pi
DEG2RAD = math.pi/180.0

category = 'optics'


class DiskChopper(StochasticComponentBase):

    def __init__(
            self, name,
            theta_0, yheight, radius, nu, nslit, isfirst=False, n_pulse=0,
            jitter=0, abs_out=True, delay=0, phase=0, xwidth=0,
            **kwargs):
        """
        Models a disc chopper with nslit identical slits that are symmetrically distributed on the disc.
        
        Based on the Component from McStas: https://mcstas.org/download/components/3.2/optics/DiskChopper.comp 

        Parameters:
        name (str): the name of this component
        theta_0 (deg): angular width of the slits
        yheight(m): slit height (if = 0, then equal to radius). Auto centerting of the beam at half height.
        radius (m): radius of the disc
        nu (Hz): frequency of the chopper, omega = 2*PI*nu (+/- defines direction of rotation)
        nslit (number): number of slits, symettrically distributed around disk

        Optional Parameters:
        isfist (False/True): Set to True for the first chopper position in a cw source (it then spreads the neutron time distribution)
        n_pulse (number): Number of pulses, only used when isfirst=True
        jitter (s): Jitter in the time phase
        abs_out (False/True): Whether to abosrb neutrons that hit outside of the chopper radius
        delay (s): Time 'delay'
        phase (deg): Angular 'delay'. NOTE: overrides delay
        xwidth (m): Horizontal slit width opening at beam center. 
        """
        super().__init__(__class__, **kwargs)

        self.name = name

        # check if yheight is 0. If so, assumes full opening
        if yheight == 0:
            self.height = radius
        else:
            self.height = yheight

        # radius at beam center
        delta_y = radius - 0.5 * self.height

        omega = 2.0*math.pi*nu

        if xwidth > 0 and theta_0 == 0 and radius > 0:
            theta_0 = 2 * RAD2DEG * math.asin(xwidth/(2.0 * delta_y))

        # check for valid configurations
        if nslit <= 0 or theta_0 <= 0 or radius <= 0:
            raise RuntimeError(
                "DiskChopper '{}': nslit, theta_0, and radius must be > 0".format(name))

        if nslit * theta_0 >= 360.0:
            raise RuntimeError(
                "DiskChopper '{}': nslit * theta_0 exceeds 2*PI (got {})".format(name, theta_0*2.0*math.pi))

        if yheight != 0 and yheight > radius:
            raise RuntimeError("DiskChopper '{}': yheight must be < radius".format(name))


        if isfirst and n_pulse <= 0:
            raise RuntimeError("DiskChopper '{}': wrong First chopper pulse number (n_pulse = {})".format(name, n_pulse))

        if omega == 0:
            omega = 1.0e-15
            raise RuntimeWarning(
                "DiskChopper '{}': WARNING - chopper frequency is 0!".format(name))

        if abs_out == False:
            raise RuntimeWarning(
                "DiskChopper '{}': WARNING - chopper will NOT absorb neutrons outside radius {}!".format(name, radius))

        # convert to radians
        theta_0 *= DEG2RAD

        # calculate delay using phase, or phase using delay
        if phase > 0.0:
            if delay > 0.0:
                raise RuntimeWarning(
                    "DiskChopper - '{}: WARNING - both delay AND phase are specified.. ignoring delay and using phase..")
            phase *= DEG2RAD
            # delay should always be a "delay" - regardless of rotation direction (i.e, negative omega)
            delay = phase / math.fabs(omega)
        else:
            phase = delay * omega

        # Time from opening of slit to next slit opening
        time_g = 2.0 * math.pi / math.fabs(omega) / float(nslit)

        # How long neutrons can pass the chopper at a single point
        time_open = theta_0 / math.fabs(omega)

        if xwidth == 0:
            xwidth = 2.0 * delta_y * math.sin(0.5 * theta_0)

        self.propagate_params = (np.array([abs_out, delta_y, radius, self.height,
                                           omega, time_open, delay, jitter, n_pulse, time_g, nslit, theta_0, isfirst]),)

        print(self.propagate_params)
        # Aim neutrons toward the slit to cause JIT compilation.
        import mcni
        neutrons = mcni.neutron_buffer(1)
        neutrons[0] = mcni.neutron(r=(0, 0, -1), v=(0, 0, 1), prob=1, time=0)

        self.process(neutrons)

    @cuda.jit(void(int64, xoroshiro128p_type[:], NB_FLOAT[:], NB_FLOAT[:]), device=True)
    def propagate(threadindex, rng_states, neutron, param_arr):
        abs_out, delta_y, radius, height, omega, time_open, delay, jitter, n_pulse, time_g, nslit, theta_0, isfirst = param_arr

        # Propagate to Z = 0
        x, y, z, t = prop_z0(neutron)

        yprime = y + delta_y

        # check if neutron is outside vertical slit range. Absorb if abs_out = True
        if abs_out and (x*x + yprime*yprime) > radius*radius:
            absorb(neutron)
            return

        # check if the neutron hits inner solid part of chopper? (case of yheight!=radius)
        if (x*x + yprime*yprime) < (radius - height)*(radius - height):
            absorb(neutron)
            return

        jitter_factor = jitter * randnorm(rng_states, threadindex)

        if isfirst:
            # all events are put in the transmitted time frame
            pulse_factor = math.floor(
                n_pulse * rand01(rng_states, threadindex)) * time_g if n_pulse > 1 else 0.0

            t = math.atan2(x, yprime) / omega + 0.5*time_open * \
                randpm1(rng_states, threadindex) + \
                delay + jitter_factor + pulse_factor

            # correction: chopper slits transmission opening/full disk
            neutron[-1] *= nslit * theta_0 / 2.0 / math.pi
        else:

            toff = math.fabs(t - math.atan2(x, yprime) /
                                  omega - delay - jitter_factor)

            # absorb if neutron hits outside of the slit
            if math.fmod(toff + 0.5 * time_open, time_g) > time_open:
                absorb(neutron)
                return

        # SCATTER
        neutron[:3] = x, y, z
        neutron[-2] = t


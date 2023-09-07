#!/usr/bin/env python

import numpy as np
from numba import cuda, void, int64
import math
import re

from ..StochasticComponentBase import StochasticComponentBase
from ...neutron import absorb, prop_z0
from ...random import randnorm
import warnings

from ...config import get_numba_floattype
from numba.cuda.random import xoroshiro128p_type

NB_FLOAT = get_numba_floattype()
RAD2DEG = 180./math.pi
DEG2RAD = math.pi/180.0

category = 'optics'


class MultiDiskChopper(StochasticComponentBase):

    def __init__(
            self, name,
            slit_width="10_20", slit_center="0_180", nslits=2, radius=0.375, delta_y=-0.3, nu=0.0, isfirst=False,
            jitter=0, abs_out=True, delay=0, phase=0,
            **kwargs):
        """
        Models a disc chopper with nslit identical slits that are symmetrically distributed on the disc.
        
        Based on the Component from McStas: https://mcstas.org/download/components/3.2/optics/DiskChopper.comp 

        Parameters:
        name (str): the name of this component
        slit_width (str): angular (deg) width of the slits, given as a list in a string separated by space (' '), comma (','), underscore ('_'), or semicolon (';'). Example: "0;20;90;135;270"
        slit_center (str): angular (deg) position of the slits (specified same as slit_width)
        nslits (int): Number of slits to read from slit_width and slit_center.
        radius (m): outer radius of the disk
        delta_y (m): y-position of the chopper rotation axis.
        nu (Hz): frequency of the chopper, omega = 2*PI*nu (+/- defines direction of rotation)

        Optional Parameters:
        isfist (False/True): Set to True for the first chopper position in a cw source (it then spreads the neutron time distribution)
        jitter (s): Jitter in the time phase
        abs_out (False/True): Whether to abosrb neutrons that hit outside of the chopper radius
        delay (s): Time 'delay'
        phase (deg): Angular 'delay'. NOTE: phase and delay are cumulative and can both be specified, in contrast to the DiskChopper component.        
        """
        super().__init__(__class__, **kwargs)

        self.name = name

        phase = math.remainder(phase, 360.0) * DEG2RAD
        omega = 2.0*math.pi*nu # rad/s

        if omega == 0:
            warnings.warn(
                "MultiDiskChopper '{}': WARNING - chopper frequency is 0!".format(name))
            omega = 1.0e-15

        # check for valid configurations
        if nslits <= 0:
            raise RuntimeError(
                "MultiDiskChopper '{}': nslits must be > 0".format(name))
        
        dslit_center = np.asarray(re.split('[ ,;_]', slit_center), dtype=float)
        dhslit_width = np.asarray(re.split('[ ,;_]', slit_width), dtype=float)

        print(" DSLIT_CENTER (IN) = {}".format(dslit_center))
        print(" DSLIT_WIDTH  (IN) = {}".format(dhslit_width))

        assert len(dslit_center) == nslits
        assert len(dhslit_width) == nslits

        for i in range(nslits):
            if dslit_center[i] < 0:
                while dslit_center[i] < 0:
                    dslit_center[i] += 360.0

            if dslit_center[i] >= 360.0:
                while dslit_center[i] >= 360.0:
                    dslit_center[i] -= 360.0

            # dhslit_width: HALF slit width
            dhslit_width[i] *= 0.5

            if dhslit_width[i] <= 0.0:
                raise RuntimeError("MultiDiskChopper '{}': Slit no {} has nonpositive width!".format(name, i))

            dslit_center[i] *= DEG2RAD
            dhslit_width[i] *= DEG2RAD

        print(" DSLIT_CENTER (RAD) = {}".format(dslit_center))
        print(" DSLIT_WIDTH  (RAD) = {}".format(dhslit_width))

        # calculate delay using phase, or phase using delay
        if phase:
            if delay:
                warnings.warn(
                    "MultiDiskChopper - '{}: both delay AND phase are specified.. adding them both together".format(name))
            phase -= delay*omega
            delay = -phase / omega
        else:
            phase = delay * omega

        # Time for 1 revolution
        T = 2.0 * math.pi / math.fabs(omega)

        # generate arrays of times t0 and t1
        # t0 = np.zeros(nslits, dtype=float)
        # t1 = np.zeros(nslits, dtype=float)

        # TODO: finish implementation for isfirst
        if isfirst:
            raise RuntimeError("isfirst = True not implemented")

        self.propagate_params = (np.array([abs_out, delta_y, radius, nslits,  omega, delay, jitter, T, isfirst]), dslit_center, dhslit_width)

        print(self.propagate_params)
        # Aim neutrons toward the slit to cause JIT compilation.
        import mcni
        neutrons = mcni.neutron_buffer(1)
        neutrons[0] = mcni.neutron(r=(0, 0, -1), v=(0, 0, 1), prob=1, time=0)

        self.process(neutrons)

    @cuda.jit(void(int64, xoroshiro128p_type[:], NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:]), device=True)
    def propagate(threadindex, rng_states, neutron, param_arr, dslit_center, dhslit_width):
        abs_out, delta_y, radius, nslits,  omega, delay, jitter, T, isfirst = param_arr

        # Propagate to Z = 0
        x, y, z, t = prop_z0(neutron)

        if delta_y > 0.0:
            # "anormal" case, chopper above guide
            # mirror coordinate system
            xprime = -x
            yprime = -y + delta_y
        else:
            # "normal" case, chopper below guide
            xprime = x
            yprime = y - delta_y

        # check if neutron is outside vertical slit range. Absorb if abs_out = True, scatter otherwise
        if xprime*xprime + yprime*yprime > radius*radius:
            if abs_out:
                absorb(neutron)
                return
            else:
                # SCATTER
                neutron[:3] = x, y, z
                neutron[-2] = t
                return
        else:
            if isfirst:
                # TODO: finish implementation
                absorb(neutron)
                return
            else:
                jitter_factor = jitter * randnorm(rng_states, threadindex)

                # where does the neutron hit the disk?
                phi = math.atan2(xprime, yprime) + omega * (t - delay - jitter_factor)

                # does the neutron hit one of the slits?
                islit = int(0)
                while islit < nslits:
                    delta_slit = math.fabs( math.remainder( phi - dslit_center[islit], math.pi * 2.0))
                    if delta_slit < dhslit_width[islit]:
                        # SCATTER
                        neutron[:3] = x, y, z
                        neutron[-2] = t
                        return
                    islit += int(1)

                absorb(neutron)
                return

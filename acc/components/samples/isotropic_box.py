# -*- python -*-
#
# Jiao Lin <jiao.lin@gmail.com>
#

import numpy as np
from numba import cuda, void, int64
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type
from math import sqrt, exp

from .SampleBase import SampleBase
from ...neutron import absorb, prop_dt_inplace
from ...geometry.onbox import cu_device_intersect_box
from ...kernels.isotropic import S as kernel

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()


category = 'samples'
class IsotropicBox(SampleBase):

    def __init__(
            self, name,
            xwidth, yheight, zthickness,
            mu, sigma,
    ):
        """
        Initialize the isotropicbox component.

        Parameters:
        name (str): the name of this component
        xwidth (m): width
        yheight (m): height
        zthickness (m): thickness
        mu (1/m): inverse absorption length
        sigma (1/m): inverse scattering length
        """
        self.name = name
        self.propagate_params = (
            np.array([xwidth, yheight, zthickness, mu, sigma]),
        )

        # Aim neutrons toward the sample to cause JIT compilation.
        import mcni
        neutrons = mcni.neutron_buffer(1)
        neutrons[0] = mcni.neutron(r=(0, 0, -1), v=(0, 0, 1), prob=1, time=0)
        self.process(neutrons)

    @cuda.jit(void(
        int64, xoroshiro128p_type[:],
        NB_FLOAT[:],
        NB_FLOAT[:]
    ), device=True)
    def propagate(
            threadindex, rng_states,
            neutron, param_arr
    ):
        xwidth, yheight, zthickness, mu, sigma = param_arr
        x, y, z, vx, vy, vz = neutron[:6]
        t1, t2 = cu_device_intersect_box(x,y,z, vx,vy,vz, xwidth, yheight, zthickness)
        if t2<=0:
            # no interception
            return
        elif t2>0:
            if t1 <=0:
                # starting from current position
                t1 = 0
        else:
            # NaN. no interception
            return
        if t1>0:
            # propagate to surface
            prop_dt_inplace(neutron, t1)
            t2 = t2-t1
        # t2 is now the time to get to the exiting surface
        dt = xoroshiro128p_uniform_float32(rng_states, threadindex)*(t2)
        v = sqrt(vx*vx+vy*vy+vz*vz)
        dist = v*dt
        fulllen = v*t2
        atten = exp( -(mu/v*2200+sigma) * dist )
        prob = sigma * fulllen * atten
        # prob *= sum_of_weights/m_weights.scattering;
        neutron[-1] *= prob
        # neutron is now at the location for scattering
        prop_dt_inplace( neutron, dt )
        # kernel
        kernel(threadindex, rng_states, neutron)
        # ev.probability *= packing_factor;
        if neutron[-1] <=0:
            absorb(neutron)
            return
        # find exiting time
        x, y, z, vx, vy, vz = neutron[:6]
        t1, t2 = cu_device_intersect_box(x,y,z, vx,vy,vz, xwidth, yheight, zthickness)
        atten2 = exp( -(mu/v*2200+sigma) * v * t2 )
        neutron[-1] *= atten2
        return

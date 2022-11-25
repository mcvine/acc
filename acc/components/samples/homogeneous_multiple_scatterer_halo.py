# -*- python -*-
#
# Cole Kendrick <kendrickcj@ornl.gov>
#

import numpy as np, numba
from numba import cuda, void, int64, int32
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type
from math import sqrt, exp, pi, cos, sin

from ... import test
from .SampleBase import SampleBase
from ...neutron import absorb, prop_dt_inplace
from ...geometry.arrow_intersect import max_intersections
from .homogeneous_single_scatterer import calc_time_to_point_of_scattering, \
    _calc_time_to_point_of_scattering_impl, total_time_in_shape, time_to_enter

from numba.core import config
if not config.ENABLE_CUDASIM:
    from numba.cuda.compiler import Dispatcher, DeviceFunction

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()


category = 'samples'

@cuda.jit(device=True)
def dummy_absorb(neutron):
    return

def factory(shape, kernel):
    """
    Usage:
    * create a new python module with code to define a new scatterer class
      that inherits from a base class created using this factory method.
      see tests/components/samples/HSS_isotropic_sphere.py for an example
      - ...load shape and kernel...
      - HSSbase = factory(shape = shape, kernel = None)
      - class HSS(HSSbase): pass
    """
    from ...geometry import arrow_intersect
    intersect = arrow_intersect.arrow_intersect_func_factory.render(shape)
    locate = arrow_intersect.locate_func_factory.render(shape)
    from . import getAbsScttCoeffs
    mu, sigma = getAbsScttCoeffs(kernel)
    from ...kernels import scatter_func_factory
    scatter, calc_scattering_coeff, absorb = scatter_func_factory.render(kernel)
    if calc_scattering_coeff is None:
        @cuda.jit(device=True)
        def calc_scattering_coeff(neutron):
            return sigma
    if absorb is None:
        absorb = dummy_absorb

    # sets the number of neutrons scattered by this component
    # has to be defined outside of the class so it is visible in propagate
    N = 100

    class HomogeneousMultipleScattererTest(SampleBase):

        is_multiscattering = True
        NUM_MULTIPLE_SCATTER = N

        def __init__(self, name):
            """
            Initialize the isotropicbox component.

            Parameters:
            name (str): the name of this component
            """
            self.name = name
            self.propagate_params = ()

            # Aim neutrons toward the sample to cause JIT compilation.
            import mcni
            neutrons = mcni.neutron_buffer(1)
            neutrons[0] = mcni.neutron(r=(0, 0, -1), v=(0, 0, 1), prob=1, time=0)
            self.process(neutrons)

        @cuda.jit(
            int32(int64, xoroshiro128p_type[:], NB_FLOAT[:, :], NB_FLOAT[:],
        ) , device=True)
        def propagate(threadindex, rng_states, out_neutrons, neutron):
            x, y, z, vx, vy, vz = neutron[:6]
            # loc = locate(x,y,z)
            ts = cuda.local.array(max_intersections, dtype=numba.float64)
            ninter = intersect(x,y,z, vx,vy,vz, ts)
            if ninter < 2: return 0
            if ts[ninter-1] <= 0: return 0
            rand = xoroshiro128p_uniform_float32(rng_states, threadindex)
            total_time_in_shape1, total_time_travel_to_scattering_point, time_travelled_in_shape_to_scattering_point = calc_time_to_point_of_scattering(ts, ninter, rand)
            dt = total_time_travel_to_scattering_point
            if dt<=0: return 0
            # propagate to scattering point
            prop_dt_inplace(neutron, dt)
            # calc attenuation
            v = sqrt(vx*vx+vy*vy+vz*vz)
            dist = v*time_travelled_in_shape_to_scattering_point
            fulllen = v*total_time_in_shape1
            sigma = calc_scattering_coeff(neutron)
            atten = exp( -(mu/v*2200+sigma) * dist )
            prob = fulllen * atten  # Xsigma is now part of scatter method
            # prob *= sum_of_weights/m_weights.scattering;
            neutron[-1] *= prob
            # kernel
            scatter(threadindex, rng_states, neutron)
            # ev.probability *= packing_factor;
            if neutron[-1] <=0:
                absorb(neutron)
                return 0
            # find exiting time
            x, y, z, vx, vy, vz = neutron[:6]
            ninter = intersect(x,y,z, vx,vy,vz, ts)
            dt3 = total_time_in_shape(ts, ninter)
            sigma = calc_scattering_coeff(neutron)
            atten2 = exp( -(mu/v*2200+sigma) * v * dt3 )
            neutron[-1] *= atten2

            # set multiple scattering outputs
            for j in range(N):
                # TODO: fix this - cannot do out_neutrons[j] = neutron[:]
                for k in range(10):
                    out_neutrons[j][k] = neutron[k]
                if j > 1:
                    # simple test that randomly scatters neutrons into a circle
                    r = 0.01
                    rand = xoroshiro128p_uniform_float32(rng_states, threadindex) - 0.5
                    out_neutrons[j][0] += r * cos(2*pi*rand)
                    out_neutrons[j][1] += r * sin(2*pi*rand)
            return N

    return HomogeneousMultipleScattererTest


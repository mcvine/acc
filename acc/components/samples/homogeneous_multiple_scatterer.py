# -*- python -*-
#
# Jiao Lin <jiao.lin@gmail.com>
#

import numpy as np, numba
from numba import cuda, void, int64
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type
from math import sqrt, exp

from ... import test
from .SampleBase import SampleBase
from ...neutron import absorb, prop_dt_inplace, clone
from ... import vec3
from ...geometry.arrow_intersect import max_intersections
from ...geometry.location import inside

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
    from ...geometry import propagation
    propagators = propagation.makePropagateMethods(intersect, locate)
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

    # propagate methods
    propagate_to_next_incident_surface = propagators['propagate_to_next_incident_surface']
    propagate_to_next_exiting_surface = propagators['propagate_to_next_exiting_surface']
    tof_before_exit = propagators['tof_before_exit']

    @cuda.jit(
        void(int64, xoroshiro128p_type[:], NB_FLOAT[:, :], NB_FLOAT[:]
    ) , device=True)
    def _interactM1(threadindex, rng_states, out_neutrons, neutron):
        """neutron interact with the scatterer once and generate scattered and transmitted neutrons.
        also run the absorption mechanism
        """
        x, y, z, vx, vy, vz = neutron[:6]
        loc = locate(x,y,z)
        if loc!= inside:
            propagate_to_next_incident_surface(neutron)
        tof = tof_before_exit(neutron)
        v = sqrt(vx*vx+vy*vy+vz*vz)
        distance = tof*v
        sigma = calc_scattering_coeff(neutron)
        mu1 = mu/v*2200
        transmission_prob = exp(-(mu1+sigma) * distance)
        absorption_prob = (1-transmission_prob)*(mu1/(mu1+sigma))
        # transmission
        transmitted = out_neutrons[0]
        clone(neutron, transmitted)
        propagate_to_next_exiting_surface(transmitted)
        transmitted[-1] *= transmission_prob
        # absorption
        temp = out_neutrons[1]
        clone(neutron, temp)
        temp[-1] *= absorption_prob
        absorb(temp)
        # scattering
        rand = xoroshiro128p_uniform_float32(rng_states, threadindex)
        x = distance*rand
        prob = distance * exp( -(mu1+sigma) * x );
        scattered = out_neutrons[1]
        clone(neutron, scattered)
        scattered[-1] *= prob
        prop_dt_inplace(scattered, x/v)
        scatter(threadindex, rng_states, scattered)
        # packing_factor
        return

    class HomogeneousMultipleScatterer(SampleBase):

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
            void(int64, xoroshiro128p_type[:], NB_FLOAT[:]
        ) , device=True)
        def propagate(threadindex, rng_states, in_neutron):
            x, y, z, vx, vy, vz = in_neutron[:6]
            return

        @cuda.jit(
            void(int64, xoroshiro128p_type[:], NB_FLOAT[:], NB_FLOAT[:, :]
        ) , device=True)
        def propagateM(threadindex, rng_states, in_neutron, out_neutrons):
            x, y, z, vx, vy, vz = in_neutron[:6]
            return
    HomogeneousMultipleScatterer._interactM1 = _interactM1
    return HomogeneousMultipleScatterer

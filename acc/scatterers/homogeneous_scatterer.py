# -*- python -*-
#
# Jiao Lin <jiao.lin@gmail.com>
#

import numpy as np, numba
from numba import cuda, void, int64
from mcvine.acc._numba import xoroshiro128p_uniform_float32
from math import sqrt, exp

from .interaction_types import absorption, scattering, none
from .. import test
from ..vec3 import (
    subtract as v3_subtract,
    length as v3_length,
    add as v3_add,
    add_inplace as v3_add_inplace,
    scale as v3_scale,
    copy as v3_copy,
)
from ..neutron import absorb, prop_dt_inplace
from ..geometry.arrow_intersect import max_intersections
from ..geometry.locate import inside, outside, onborder
from ..geometry.propagation import makePropagateMethods


from numba.core import config
if not config.ENABLE_CUDASIM:
    from numba.cuda.compiler import Dispatcher, DeviceFunction

from ..config import get_numba_floattype
NB_FLOAT = get_numba_floattype()


def factory(shape, kernel, mcweights, packing_factor):
    w_absorption, w_scattering, w_transmission = mcweights
    from ..geometry import arrow_intersect
    intersect = arrow_intersect.arrow_intersect_func_factory.render(shape)
    locate = arrow_intersect.locate_func_factory.render(shape)
    propagate_methods = makePropagateMethods(intersect, locate)
    propagate_to_next_incident_surface = propagate_methods['propagate_to_next_incident_surface']
    propagate_to_next_exiting_surface = propagate_methods['propagate_to_next_exiting_surface']
    tof_before_first_exit = propagate_methods['tof_before_first_exit']
    forward_distance_in_shape = propagate_methods["forward_distance_in_shape"]
    from ..kernels import scatter_func_factory
    scatter, calc_scattering_coeff, absorb, calc_absorption_coeff = scatter_func_factory.render(kernel)
    def calculate_attenuation(neutron, end):
        length = forward_distance_in_shape(neutron, end)
        sigma = calc_scattering_coeff(neutron) * packing_factor
        mu = calc_absorption_coeff(neutron) * packing_factor
        return exp( - (mu+sigma) * length )

    def _interact_path1(threadindex, rng_states, neutron, tmp_neutron):
        x, y, z, vx, vy, vz = neutron[:6]
        loc = locate(x,y,z)
        if loc != inside:
            # propagate to the front surface
            propagate_to_next_incident_surface(neutron)
        # distance to travel across and leave for the first time
        tof = tof_before_first_exit(neutron)
        velocity = sqrt(vx*vx+vy*vy+vz*vz)
        distance = velocity * tof
        # scattering and absorption coeffs
        sigma = calc_scattering_coeff(neutron)
        mu = calc_absorption_coeff(neutron)
        # probability of three interaction types happening
        transmission_prob = exp( -(mu+sigma)*distance )
        absorption_prob = (1-transmission_prob)*(mu/(mu+sigma))
        # scattering_prob = (1-transmission_prob)*(sigma/(mu+sigma))
        #
        # toss a dice and decide whether we should do transmission, absorption,
        # or scattering
        transmission_mark = w_transmission
        absorption_mark = transmission_mark + w_absorption
        sum_of_weights = absorption_mark + w_scattering
        r = xoroshiro128p_uniform_float32(rng_states, threadindex) * sum_of_weights
        if (r < transmission_mark) :
            # transmission
            propagate_to_next_exiting_surface( neutron )
            neutron[-1] *= transmission_prob * (sum_of_weights/w_transmission)
            return none
        if (r >= transmission_mark and r < absorption_mark ):
            # absorption
            x = xoroshiro128p_uniform_float32(rng_states, threadindex) * distance
            prob = mu * distance * exp( -(mu+sigma) * x )
            neutron[-1] *= prob * (sum_of_weights/w_absorption)
            prop_dt_inplace( neutron, x/velocity )
            absorb( threadindex, rng_states, neutron )
            neutron[-1] = -1
            return absorption
        # scattering
        x = xoroshiro128p_uniform_float32(rng_states, threadindex) * distance
        atten = exp( -(mu+sigma) * x )
        prob = distance * atten
        prob *= sum_of_weights/w_scattering
        neutron[-1] *= prob
        prop_dt_inplace( neutron, x/velocity )
        scatter( threadindex, rng_states, neutron )
        neutron[-1] *= packing_factor
        if neutron[-1] <=0:
            return absorption
        for i in range(10):
            tmp_neutron[i] = neutron[i]
        propagate_to_next_exiting_surface( neutron )
        atten2 = calculate_attenuation( tmp_neutron, neutron[:3] );
        neutron[-1] *= atten2;
        return scattering
    if test.USE_CUDASIM:
        def interact_path1(threadindex, rng_states, neutron):
            tmp_neutron = np.zeros(10, dtype=float)
            _interact_path1(threadindex, rng_states, neutron, tmp_neutron)
    else:
        def interact_path1(threadindex, rng_states, neutron):
            tmp_neutron = cuda.local.array(10, dtype=numba.float64)
            _interact_path1(threadindex, rng_states, neutron, tmp_neutron)
    return dict(
        interact_path1 = interact_path1,
        calculate_attenuation = calculate_attenuation,
    )

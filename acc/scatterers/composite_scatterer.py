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
from ..neutron import absorb, prop_dt_inplace, clone
from ..geometry.arrow_intersect import max_intersections
from ..geometry.locate import inside, outside, onborder
from ..geometry.propagation import makePropagateMethods


from numba.core import config
if not config.ENABLE_CUDASIM:
    from numba.cuda.compiler import Dispatcher, DeviceFunction

from ..config import get_numba_floattype
NB_FLOAT = get_numba_floattype()


def factory_3(composite):
    # methods to deal with the whole shape of the composite
    shape = composite.shape()
    from ..geometry import arrow_intersect
    intersect = arrow_intersect.arrow_intersect_func_factory.render(shape)
    locate = arrow_intersect.locate_func_factory.render(shape)
    propagate_methods = makePropagateMethods(intersect, locate)
    propagate_to_next_incident_surface = propagate_methods['propagate_to_next_incident_surface']
    propagate_to_next_exiting_surface = propagate_methods['propagate_to_next_exiting_surface']
    tof_before_first_exit = propagate_methods['tof_before_first_exit']
    forward_distance_in_shape = propagate_methods["forward_distance_in_shape"]
    # elements
    elements = composite.elements()
    nelements = len(elements)
    assert nelements == 3
    # find_1st_hit
    shapes = [e.shape() for e in elements]
    from mcvine.acc.geometry.composite import createMethods_3, make_find_1st_hit
    methods = createMethods_3(shapes)
    find_1st_hit = make_find_1st_hit(**methods)
    # methods for element scatterers
    from . import scatter_func_factory
    #element_scatter_methods = [scatter_func_factory.render(e) for e in elements]
    element_scatter_methods = []
    for e in elements:
        m = scatter_func_factory.render(e)
        element_scatter_methods.append(m)
    element0_interact_path1 = element_scatter_methods[0]['interact_path1']
    element1_interact_path1 = element_scatter_methods[1]['interact_path1']
    element2_interact_path1 = element_scatter_methods[2]['interact_path1']
    element0_calculate_attenuation = element_scatter_methods[0]['calculate_attenuation']
    element1_calculate_attenuation = element_scatter_methods[1]['calculate_attenuation']
    element2_calculate_attenuation = element_scatter_methods[2]['calculate_attenuation']

    def _interact_path1(threadindex, rng_states, neutron, tmp_neutron):
        x,y,z = neutron[:3]
        vx,vy,vz = neutron[3:6]
        scatterer_index = find_1st_hit(x,y,z, vx,vy,vz)
        if scatterer_index < 0 or scatterer_index > nelements:
            propagate_to_next_exiting_surface(neutron)
            return none
        clone(neutron, tmp_neutron)
        # global2local(tmp_neutron, scatterer_index)
        interaction = element_interact_path1(threadindex, rng_states, tmp_neutron, scatterer_index)
        # local2global(tmp_neutron, scatterer_index)
        clone(tmp_neutron, neutron)
        if interaction == absorption:
            absorb(neutron)
            return interaction
        # nothing happened
        if interaction == none:
            # if neutron is not inside, we are done
            if locate(neutron)!=inside:
                return interaction
            # otherwise, interact again
            return _interact_path1(threadindex, rng_states, neutron, tmp_neutron)
        # interaction must be scatter
        clone(neutron, tmp_neutron)
        # propagate to surface if necessary
        if locate(*neutron[:3]) == inside:
            propagate_to_next_exiting_surface(neutron)
        # apply attenuation
        att = calculate_attenuation(tmp_neutron, neutron[:3])
        neutron[-1] *= att
        return scattering

    def _calculate_attenuation(neutron, end, tmp_neutron):
        ret = 1.0
        for i in range(nelements):
            clone(neutron, tmp_neutron)
            # global2local(tmp_neutron, i)
            ret *= element_calculate_attenuation(neutron, end, i)
        return ret

    def element_interact_path1(threadindex, rng_states, neutron, element_index):
        if element_index == 0:
            return element0_interact_path1(threadindex, rng_states, neutron)
        if element_index == 1:
            return element1_interact_path1(threadindex, rng_states, neutron)
        if element_index == 2:
            return element2_interact_path1(threadindex, rng_states, neutron)

    def element_calculate_attenuation(neutron, end, element_index):
        if element_index == 0:
            return element0_calculate_attenuation(neutron, end)
        if element_index == 1:
            return element1_calculate_attenuation(neutron, end)
        if element_index == 2:
            return element2_calculate_attenuation(neutron, end)

    if test.USE_CUDASIM:
        def interact_path1(threadindex, rng_states, neutron):
            tmp_neutron = np.zeros(10, dtype=float)
            return _interact_path1(threadindex, rng_states, neutron, tmp_neutron)
        def calculate_attenuation(neutron, end):
            tmp_neutron = np.zeros(10, dtype=float)
            return _calculate_attenuation(neutron, end, tmp_neutron)
    else:
        def interact_path1(threadindex, rng_states, neutron):
            tmp_neutron = cuda.local.array(10, dtype=numba.float64)
            return _interact_path1(threadindex, rng_states, neutron, tmp_neutron)
        def calculate_attenuation(neutron, end):
            tmp_neutron = cuda.local.array(10, dtype=numba.float64)
            return _calculate_attenuation(neutron, end, tmp_neutron)
    return dict(
        interact_path1 = interact_path1,
        calculate_attenuation = calculate_attenuation,
    )

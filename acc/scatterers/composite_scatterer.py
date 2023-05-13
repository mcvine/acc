# -*- python -*-
#
# Jiao Lin <jiao.lin@gmail.com>
#

from math import sqrt, exp
import numpy as np, numba
from numba import cuda
from numba.core import config

from .. import test
from .._numba import xoroshiro128p_uniform_float32
from .interaction_types import absorption, scattering, none
from ..neutron import absorb, prop_dt_inplace, clone
from ..geometry.locate import inside, outside, onborder


from ..config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

def _createRayTracingMethods(composite):
    """create ray-tracing methods to deal with the overall shape of the composite
    """
    shape = composite.shape()
    from ..geometry import arrow_intersect
    intersect = arrow_intersect.arrow_intersect_func_factory.render(shape)
    locate = arrow_intersect.locate_func_factory.render(shape)
    del shape
    from ..geometry.propagation import makePropagateMethods
    propagate_methods = makePropagateMethods(intersect, locate)
    propagate_to_next_exiting_surface = propagate_methods['propagate_to_next_exiting_surface']
    del propagate_methods
    # find_1st_hit
    elements = composite.elements()
    shapes = [e.shape() for e in elements]
    from mcvine.acc.geometry.composite import get_find_1st_hit
    find_1st_hit = get_find_1st_hit(shapes)
    return dict(
        locate = locate,
        intersect = intersect,
        find_1st_hit = find_1st_hit,
        propagate_to_next_exiting_surface = propagate_to_next_exiting_surface,
    )

def factory(composite):
    elements = composite.elements()
    N = len(elements)
    assert N==3
    from .composite_3 import createHelperMethodsForScatter
    element_methods = createHelperMethodsForScatter(composite)
    element_interact_path1 = element_methods["element_interact_path1"]
    element_calculate_attenuation = element_methods["element_calculate_attenuation"]
    rt_methods = _createRayTracingMethods(composite)
    find_1st_hit = rt_methods['find_1st_hit']
    locate = rt_methods['locate']
    propagate_to_next_exiting_surface = rt_methods['propagate_to_next_exiting_surface']
    @cuda.jit(device=True)
    def _interact_path1(threadindex, rng_states, neutron, tmp_neutron):
        x,y,z = neutron[:3]
        vx,vy,vz = neutron[3:6]
        scatterer_index = find_1st_hit(x,y,z, vx,vy,vz)
        if scatterer_index < 0 or scatterer_index > N:
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
            x,y,z = neutron[:3]
            if locate(x,y,z)!=inside:
                return interaction
            # otherwise, interact again
            # return _interact_path1(threadindex, rng_states, neutron, tmp_neutron)
            return interaction
        # interaction must be scatter
        clone(neutron, tmp_neutron)
        # propagate to surface if necessary
        x,y,z = neutron[:3]
        if locate(x,y,z) == inside:
            propagate_to_next_exiting_surface(neutron)
        # apply attenuation
        att = calculate_attenuation(tmp_neutron, neutron[:3])
        neutron[-1] *= att
        return scattering

    @cuda.jit(device=True)
    def _calculate_attenuation(neutron, end, tmp_neutron):
        ret = 1.0
        for i in range(N):
            clone(neutron, tmp_neutron)
            # global2local(tmp_neutron, i)
            ret *= element_calculate_attenuation(neutron, end, i)
        return ret

    if test.USE_CUDASIM:
        @cuda.jit(device=True, inline=True)
        def interact_path1(threadindex, rng_states, neutron):
            tmp_neutron = np.zeros(10, dtype=float)
            return _interact_path1(threadindex, rng_states, neutron, tmp_neutron)
        @cuda.jit(device=True, inline=True)
        def calculate_attenuation(neutron, end):
            tmp_neutron = np.zeros(10, dtype=float)
            return _calculate_attenuation(neutron, end, tmp_neutron)
    else:
        @cuda.jit(device=True, inline=True)
        def interact_path1(threadindex, rng_states, neutron):
            tmp_neutron = cuda.local.array(10, dtype=numba.float64)
            return _interact_path1(threadindex, rng_states, neutron, tmp_neutron)
        @cuda.jit(device=True, inline=True)
        def calculate_attenuation(neutron, end):
            tmp_neutron = cuda.local.array(10, dtype=numba.float64)
            return _calculate_attenuation(neutron, end, tmp_neutron)
    return dict(
        interact_path1 = interact_path1,
        calculate_attenuation = calculate_attenuation,
    )

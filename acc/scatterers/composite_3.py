# -*- python -*-
#
# Jiao Lin <jiao.lin@gmail.com>
#

import numpy as np, numba
from numba import cuda
from numba.core import config

from .interaction_types import none
from .. import test

def createHelperMethodsForScatter(composite):
    # elements
    elements = composite.elements()
    nelements = len(elements)
    assert nelements == 3
    # methods for element scatterers
    from . import scatter_func_factory
    element_scatter_methods = [
        scatter_func_factory.render(e)
        for e in elements
    ]
    element0_interact_path1 = element_scatter_methods[0]['interact_path1']
    element1_interact_path1 = element_scatter_methods[1]['interact_path1']
    element2_interact_path1 = element_scatter_methods[2]['interact_path1']
    element0_calculate_attenuation = element_scatter_methods[0]['calculate_attenuation']
    element1_calculate_attenuation = element_scatter_methods[1]['calculate_attenuation']
    element2_calculate_attenuation = element_scatter_methods[2]['calculate_attenuation']
    del elements, element_scatter_methods

    @cuda.jit(device=True)
    def element_interact_path1(threadindex, rng_states, neutron, element_index):
        if element_index == 0:
            return element0_interact_path1(threadindex, rng_states, neutron)
        if element_index == 1:
            return element1_interact_path1(threadindex, rng_states, neutron)
        if element_index == 2:
            return element2_interact_path1(threadindex, rng_states, neutron)
        return none

    @cuda.jit(device=True)
    def element_calculate_attenuation(neutron, end, element_index):
        if element_index == 0:
            return element0_calculate_attenuation(neutron, end)
        if element_index == 1:
            return element1_calculate_attenuation(neutron, end)
        if element_index == 2:
            return element2_calculate_attenuation(neutron, end)
        return 1.0

    return dict(
        element_interact_path1 = element_interact_path1,
        element_calculate_attenuation = element_calculate_attenuation,
    )

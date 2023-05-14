# -*- python -*-
#
# Jiao Lin <jiao.lin@gmail.com>
#

import os
from math import sqrt, exp
import numpy as np, numba
from numba import cuda
from numba.core import config

from .. import test
from .._numba import xoroshiro128p_uniform_float32
from .interaction_types import absorption, scattering, none
from ..neutron import absorb, prop_dt_inplace, clone, is_moving
from ..geometry.locate import inside, outside, onborder


from ..config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

def factory(composite):
    elements = composite.elements()
    N = len(elements)
    mod = _importModule(N)
    element_methods = mod.createHelperMethodsForScatter(composite)
    element_interact_path1 = element_methods["element_interact_path1"]
    element_calculate_attenuation = element_methods["element_calculate_attenuation"]
    rt_methods = _createRayTracingMethods(composite)
    find_1st_hit = rt_methods['find_1st_hit']
    locate = rt_methods['locate']
    intersect = rt_methods['intersect']
    propagate_out = rt_methods['propagate_out']
    propagate_to_next_exiting_surface = rt_methods['propagate_to_next_exiting_surface']
    is_exiting = rt_methods["is_exiting"]
    is_exiting_union = rt_methods["is_exiting_union"]
    propagate_to_next_incident_surface_union = rt_methods["propagate_to_next_incident_surface_union"]

    @cuda.jit(device=True)
    def _scatter(threadindex, rng_states, neutron, tmp_neutron, tmp_ts):
        if not is_moving(neutron):
            absorb(neutron)
            return
        if is_exiting(neutron):
            return
        if is_exiting_union(neutron):
            propagate_to_next_exiting_surface(neutron)
            return
        while True:
            itype = _interact_path1(threadindex, rng_states, neutron, tmp_neutron)
            if itype == absorption:
                return
            if itype == scattering:
                x,y,z, vx,vy,vz = neutron[:6]
                Nintersect = intersect(x,y,z, vx,vy,vz, tmp_ts)
                if Nintersect == 0:
                    propagate_out(neutron)
                    return
                clone(neutron, tmp_neutron)
                propagate_out(neutron)
                att = calculate_attenuation(tmp_neutron, neutron[:3])
                neutron[-1] *= att
                return
            # itype == none
            if is_exiting(neutron):
                return
            if is_exiting_union(neutron):
                propagate_to_next_exiting_surface(neutron)
                return
            propagate_to_next_incident_surface_union(neutron)
        return

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

    from ..geometry.arrow_intersect import max_intersections
    if test.USE_CUDASIM:
        @cuda.jit(device=True, inline=True)
        def interact_path1(threadindex, rng_states, neutron):
            tmp_neutron = np.zeros(10, dtype=float)
            return _interact_path1(threadindex, rng_states, neutron, tmp_neutron)
        @cuda.jit(device=True, inline=True)
        def calculate_attenuation(neutron, end):
            tmp_neutron = np.zeros(10, dtype=float)
            return _calculate_attenuation(neutron, end, tmp_neutron)
        @cuda.jit(device=True, inline=True)
        def scatter(threadindex, rng_states, neutron):
            tmp_neutron = np.zeros(10, dtype=float)
            tmp_ts = np.zeros(max_intersections, dtype=float)
            return _scatter(threadindex, rng_states, neutron, tmp_neutron, tmp_ts)
    else:
        @cuda.jit(device=True, inline=True)
        def interact_path1(threadindex, rng_states, neutron):
            tmp_neutron = cuda.local.array(10, dtype=numba.float64)
            return _interact_path1(threadindex, rng_states, neutron, tmp_neutron)
        @cuda.jit(device=True, inline=True)
        def calculate_attenuation(neutron, end):
            tmp_neutron = cuda.local.array(10, dtype=numba.float64)
            return _calculate_attenuation(neutron, end, tmp_neutron)
        @cuda.jit(device=True, inline=True)
        def scatter(threadindex, rng_states, neutron):
            tmp_neutron = cuda.local.array(10, dtype=numba.float64)
            tmp_ts = cuda.local.array(max_intersections, dtype=numba.float64)
            return _scatter(threadindex, rng_states, neutron, tmp_neutron, tmp_ts)
    return dict(
        scatter = scatter,
        interact_path1 = interact_path1,
        calculate_attenuation = calculate_attenuation,
    )


class _Union:
    def identify(self, visitor): return visitor.onUnion(self)

def _createRayTracingMethods(composite):
    """create ray-tracing methods to deal with the overall shape of the composite
    """
    # methods regarding the overall shape
    shape = composite.shape()
    from ..geometry import arrow_intersect
    intersect = arrow_intersect.arrow_intersect_func_factory.render(shape)
    locate = arrow_intersect.locate_func_factory.render(shape)
    del shape
    from ..geometry.propagation import makePropagateMethods
    propagate_methods = makePropagateMethods(intersect, locate)
    ret = propagate_methods
    # methods regarding union of element shapes
    elements = composite.elements()
    shapes = [e.shape() for e in elements]
    union = _Union(); union.shapes = shapes
    u_intersect = arrow_intersect.arrow_intersect_func_factory.render(union)
    u_locate = arrow_intersect.locate_func_factory.render(union)
    u_propagate_methods = makePropagateMethods(u_intersect, u_locate)
    # find_1st_hit
    from mcvine.acc.geometry.composite import get_find_1st_hit
    ret.update(
        locate = locate,
        intersect = intersect,
        find_1st_hit = get_find_1st_hit(shapes),
        is_exiting_union = u_propagate_methods['is_exiting'],
        propagate_to_next_incident_surface_union = u_propagate_methods['propagate_to_next_incident_surface'],
    )
    return ret

def _importModule(N):
    mod = _makeModule(N)
    import imp
    return imp.load_source(f"scatterer_composite_{N}", mod)

def _makeModule(N, overwrite=False):
    "make a python module for cuda device methods for composite scatterer with N elements"
    modulepath = coder.getModule("composite_scatterer", N)
    if os.path.exists(modulepath) and not overwrite:
        return modulepath
    indent = 4*' '
    imports = """import os, numpy as np, numba
from numba import cuda
from numba.core import config

from mcvine.acc.scatterers.interaction_types import none
from mcvine.acc import test
""".splitlines()
    createHelperMethodsForScatter = _Coder_createHelperMethodsForScatter(N, indent)()
    lines = imports + [''] + createHelperMethodsForScatter
    with open(modulepath, 'wt') as ostream:
        ostream.write("\n".join(lines))
    return modulepath

from .._numba import coder
class _Coder_createHelperMethodsForScatter:

    def __init__(self, N, indent=4*' '):
        self.N = N
        self.indent = indent
        return

    def __call__(self):
        N, indent = self.N, self.indent
        header = f"""# elements
elements = composite.elements()
nelements = len(elements)
# methods for element scatterers
from mcvine.acc.scatterers import scatter_func_factory
element_scatter_methods = [
    scatter_func_factory.render(e)
    for e in elements
]
""".splitlines()
        interact_path1_loop = coder.unrollLoop(
            N = N,
            indent = '',
            in_loop = ["element{i}_interact_path1 = element_scatter_methods[{i}]['interact_path1']"],
        )
        calculate_attenuation_loop = coder.unrollLoop(
            N = N,
            indent = '',
            in_loop = ["element{i}_calculate_attenuation = element_scatter_methods[{i}]['calculate_attenuation']"],
        )
        footer = """
return dict(
    element_interact_path1 = element_interact_path1,
    element_calculate_attenuation = element_calculate_attenuation,
)
""".splitlines()
        body = (
            header
            + interact_path1_loop
            + calculate_attenuation_loop
            + ['']
            + self.element_interact_path1()
            + self.element_calculate_attenuation()
            + footer
        )
        add_indent = lambda lines, n: [indent*n+l for l in lines]
        return (
            ["def createHelperMethodsForScatter(composite):"]
            + add_indent(body, 1)
        )

    def element_interact_path1(self):
        N, indent = self.N, self.indent
        header = [
            "@cuda.jit(device=True)",
            "def element_interact_path1(threadindex, rng_states, neutron, element_index):"
        ]
        loop = coder.unrollLoop(
            N = N,
            indent = indent,
            in_loop = [
                "if element_index == {i}:",
                indent + "return element{i}_interact_path1(threadindex, rng_states, neutron)",
            ]
        )
        return header + loop + [indent + 'return none']

    def element_calculate_attenuation(self):
        N, indent = self.N, self.indent
        header = [
            "@cuda.jit(device=True)",
            "def element_calculate_attenuation(neutron, end, element_index):"
        ]
        loop = coder.unrollLoop(
            N = N,
            indent = indent,
            in_loop = [
                "if element_index == {i}:",
                indent + "return element{i}_calculate_attenuation(neutron, end)",
            ]
        )
        return header + loop + [indent + 'return 1.0']


